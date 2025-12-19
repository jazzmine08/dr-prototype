# app.py
# Prototype: Preprocessing + EDA + Training (3 CNN) + Ensemble + Results + Predict (APTOS + DRTiD)
# ✅ Revised: APTOS and DRTiD use the SAME preprocessing conditions + SAME deterministic train/val split
#            so training + ensemble always see identical folder layout:
#            <processed_final>/train/<0..4> and <processed_final>/val/<0..4>

from __future__ import annotations

import os
import time
import json
import base64
import traceback
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, jsonify, request, send_file, abort
from flask_socketio import SocketIO

# training modules
from training.trainer_three_cnn import run_cnn_training
from training.ensemble_runner import run_ensemble
import training.training_cnn_v2 as tcnn

# =========================================================
# CLASS LABELS (BACKEND ONLY)
# =========================================================
CLASS_LABELS_5 = {
    0: "No DR (Normal)",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

def _label_from_index(idx: int) -> str:
    try:
        idx = int(idx)
    except Exception:
        return str(idx)
    return CLASS_LABELS_5.get(idx, f"Class {idx}")


# =========================================================
# SPLIT SETTINGS (applies equally to APTOS + DRTiD)
# =========================================================
VAL_RATIO = 0.20     # 80/20 split
SPLIT_SEED = 123     # deterministic split seed

def _stable_bucket(s: str, seed: int = SPLIT_SEED) -> float:
    """
    Deterministic pseudo-random in [0,1) from string.
    Same image_id will always go to same split across runs/machines.
    """
    s = (s or "").strip()
    h = hashlib.md5(f"{seed}::{s}".encode("utf-8")).hexdigest()
    # use first 8 hex chars -> 32-bit int
    v = int(h[:8], 16)
    return (v % 1_000_000) / 1_000_000.0

def _assign_split(image_id: str, val_ratio: float = VAL_RATIO, seed: int = SPLIT_SEED) -> str:
    return "val" if _stable_bucket(image_id, seed=seed) < float(val_ratio) else "train"


# =========================================================
# PATHS + STORAGE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

DRIVE_ROOT = Path(r"G:\My Drive")
RAW_DATASETS_ROOT = DRIVE_ROOT / "dataset"

# standardized outputs (preprocessed images, runs, ensemble_runs, etc.)
PROCESSED_ROOT = DRIVE_ROOT / "dr_prototype" / "processed"
LOCAL_FALLBACK_ROOT = BASE_DIR / "dr_prototype_fallback" / "processed"

USE_LOCAL = os.getenv("DR_USE_LOCAL", "0") == "1"
LOCAL_ROOT = Path(r"X:\Improving Diabetic Retinopathy Grading Accuracy\dr_runs")
LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

if USE_LOCAL:
    STORAGE_ROOT = LOCAL_ROOT
else:
    STORAGE_ROOT = PROCESSED_ROOT if DRIVE_ROOT.exists() else LOCAL_FALLBACK_ROOT

STORAGE_ROOT.mkdir(parents=True, exist_ok=True)


# =========================================================
# DATASET REGISTRY (raw datasets)
# =========================================================
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "aptos2019": {
        "root": RAW_DATASETS_ROOT / "APTOS",
        "train_csv": RAW_DATASETS_ROOT / "APTOS" / "train.csv",
        "test_csv":  RAW_DATASETS_ROOT / "APTOS" / "test.csv",
        "train_img_dir": RAW_DATASETS_ROOT / "APTOS" / "train_images",
        "test_img_dir":  RAW_DATASETS_ROOT / "APTOS" / "test_images",
        "id_col": "id_code",
        "label_col": "diagnosis",
        "ext": ".png",
        "num_classes": 5,
    },
    "drtid": {
        "root": RAW_DATASETS_ROOT / "DRTiD",
        "train_csv": RAW_DATASETS_ROOT / "DRTiD" / "traindrtid.csv",
        "test_csv":  RAW_DATASETS_ROOT / "DRTiD" / "testdrtid.csv",
        "train_img_dir": RAW_DATASETS_ROOT / "DRTiD" / "Original_Images",
        "test_img_dir":  RAW_DATASETS_ROOT / "DRTiD" / "Original_Images",
        # ⚠️ adjust if your DRTiD CSV uses different columns
        "id_col": "id_code",
        "label_col": "diagnosis",
        "ext": ".png",
        "num_classes": 5,
    },
}

def _safe_ds(dataset_name: str) -> str:
    ds = (dataset_name or "aptos2019").strip().lower()
    return ds if ds in DATASET_REGISTRY else "aptos2019"

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    dataset_name = _safe_ds(dataset_name)
    info = DATASET_REGISTRY[dataset_name].copy()
    info["processed_base"] = STORAGE_ROOT / dataset_name
    info["processed_base"].mkdir(parents=True, exist_ok=True)
    return info

def get_output_paths(dataset_name: str) -> Dict[str, Path]:
    """
    Output per dataset:
      <STORAGE_ROOT>/<dataset>/processed_final/train/<0..4>/*.png
      <STORAGE_ROOT>/<dataset>/processed_final/val/<0..4>/*.png   ✅ (now created)
      <STORAGE_ROOT>/<dataset>/processed_final/test/*.png
    """
    base = get_dataset_info(dataset_name)["processed_base"] / "processed_final"
    return {
        "base": base,
        "train": base / "train",
        "val": base / "val",
        "test": base / "test",
    }

def ensure_class_folders(root_dir: Path, num_classes: int = 5):
    for c in range(num_classes):
        (root_dir / str(c)).mkdir(parents=True, exist_ok=True)


# =========================================================
# FLASK + SOCKETIO
# =========================================================
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = "secretkey"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.errorhandler(500)
def _err_500(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "message": "Internal Server Error",
            "path": request.path
        }), 500
    return e


# =========================================================
# PAGES
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/preprocessing")
def preprocessing_page():
    return render_template("preprocessing.html")

@app.route("/eda")
def eda_page():
    return render_template("eda.html")

@app.route("/training")
def training_page():
    return render_template("training.html")

@app.route("/ensemble")
def ensemble_page():
    return render_template("ensemble.html")

@app.route("/results")
def results_page():
    return render_template("results.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# =========================================================
# API: list datasets (dropdown)
# =========================================================
@app.route("/datasets", methods=["GET"])
def list_datasets():
    return jsonify({"datasets": list(DATASET_REGISTRY.keys())})


# =========================================================
# HELPERS (general)
# =========================================================
def _strip_ext(name: str) -> str:
    base = str(name).strip()
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        if base.lower().endswith(ext):
            return base[:-len(ext)]
    return base

def find_image(folder: Path, image_id: str, default_ext: str) -> str:
    folder = Path(folder)
    image_id = str(image_id).strip()

    exact = folder / image_id
    if exact.exists():
        return str(exact)

    base = _strip_ext(image_id)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = folder / f"{base}{ext}"
        if p.exists():
            return str(p)

    p = folder / f"{base}{default_ext}"
    if p.exists():
        return str(p)

    return ""

def img_to_base64_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def _json_error(message: str, http: int = 500, **extra):
    payload = {"status": "error", "message": message}
    payload.update(extra)
    return jsonify(payload), http


# =========================================================
# PREPROCESSING PIPELINE (same for APTOS + DRTiD)
# =========================================================
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    # Crop foreground
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img_bgr = img_bgr[y:y + h, x:x + w]

    # CLAHE in LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Gray-world-ish normalization
    img = img_bgr.astype(np.float32)
    mean = img.mean(axis=(0, 1))
    img *= mean.mean() / (mean + 1e-6)
    img_bgr = np.clip(img, 0, 255).astype(np.uint8)

    # Another CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Green channel sharpening
    g = img_bgr[:, :, 1]
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    img_bgr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)

    # base preprocessing size for storage
    return cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)

def _pp_log(msg: str):
    socketio.emit("preprocess_log", {"message": msg})

def _pp_err(msg: str):
    socketio.emit("preprocess_error", {"message": msg})

def _pp_progress(percent: int, processed: int, total: int, phase: str = ""):
    socketio.emit("preprocess_progress", {
        "percent": int(percent),
        "processed": int(processed),
        "total": int(total),
        "phase": phase or ""
    })

def _pp_done(msg: str):
    socketio.emit("preprocess_done", {"message": msg})

def preprocess_dataset_worker(dataset_name: str, limit: Optional[int] = None):
    """
    ✅ Now writes TRAIN and VAL folders deterministically for both APTOS + DRTiD:
      processed_final/train/<class>/*.png
      processed_final/val/<class>/*.png
    """
    try:
        dataset_name = _safe_ds(dataset_name)
        info = get_dataset_info(dataset_name)
        out = get_output_paths(dataset_name)
        out_train = out["train"]
        out_val = out["val"]
        out_test = out["test"]

        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)

        num_classes = int(info.get("num_classes", 5))
        ensure_class_folders(out_train, num_classes=num_classes)
        ensure_class_folders(out_val, num_classes=num_classes)

        # validate raw paths
        for k in ["train_csv", "train_img_dir", "test_csv", "test_img_dir"]:
            p = Path(info[k])
            if not p.exists():
                _pp_err(f"❌ [{dataset_name}] Path not found: {p}")
                return

        _pp_log(f"🟢 Preprocessing started [{dataset_name}]")
        _pp_log(f"📦 Output base: {out['base']}")
        _pp_log(f"🔀 Split: train={1.0-VAL_RATIO:.0%} val={VAL_RATIO:.0%} seed={SPLIT_SEED}")

        train_df = pd.read_csv(info["train_csv"])
        test_df = pd.read_csv(info["test_csv"])

        id_col = info["id_col"]
        label_col = info["label_col"]
        default_ext = info["ext"]

        if id_col not in train_df.columns or label_col not in train_df.columns:
            _pp_err(f"❌ [{dataset_name}] train csv missing columns. Need {id_col},{label_col}. Found: {list(train_df.columns)}")
            return

        test_id_col = id_col
        if test_id_col not in test_df.columns:
            candidates = [c for c in ["id_code", "image_id", "image", "filename"] if c in test_df.columns]
            if not candidates:
                _pp_err(f"❌ [{dataset_name}] test csv missing '{id_col}'. Columns: {list(test_df.columns)}")
                return
            test_id_col = candidates[0]
            _pp_log(f"⚠️ [{dataset_name}] test csv missing '{id_col}', using '{test_id_col}' instead.")

        if limit is not None and limit > 0:
            train_df = train_df.head(limit)
            test_df = test_df.head(limit)
            _pp_log(f"🧪 Limit active: train={len(train_df)} test={len(test_df)}")

        total_items = int(len(train_df) + len(test_df))
        done_items = 0

        _pp_progress(0, 0, total_items, "starting")
        socketio.sleep(0)

        # TRAIN/VAL
        processed_train = 0
        processed_val = 0
        skipped_train = 0
        t0 = time.time()

        for _, row in train_df.iterrows():
            img_id = row[id_col]
            img_id_str = str(img_id).strip()

            try:
                label = int(row[label_col])
                if label < 0 or label >= num_classes:
                    raise ValueError("label_out_of_range")
            except Exception:
                skipped_train += 1
                done_items += 1
                continue

            img_path = find_image(Path(info["train_img_dir"]), img_id_str, default_ext)
            if not img_path:
                skipped_train += 1
                done_items += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped_train += 1
                done_items += 1
                continue

            out_img = preprocess_image(img)
            out_name = f"{_strip_ext(img_id_str)}.png"

            split = _assign_split(_strip_ext(img_id_str), val_ratio=VAL_RATIO, seed=SPLIT_SEED)
            if split == "val":
                save_path = out_val / str(label) / out_name
                processed_val += 1
                phase = "val"
            else:
                save_path = out_train / str(label) / out_name
                processed_train += 1
                phase = "train"

            cv2.imwrite(str(save_path), out_img)
            done_items += 1

            if done_items % 50 == 0 or done_items == total_items:
                pct = int(round(100.0 * done_items / max(total_items, 1)))
                _pp_progress(pct, done_items, total_items, phase)
                socketio.sleep(0)

            if (processed_train + processed_val) % 300 == 0:
                _pp_log(
                    f"✅ [{dataset_name}] Train={processed_train} Val={processed_val} "
                    f"| skipped={skipped_train} | {time.time()-t0:.1f}s"
                )

        # TEST
        processed_test = 0
        skipped_test = 0
        t1 = time.time()

        for _, row in test_df.iterrows():
            img_id = row[test_id_col]
            img_id_str = str(img_id).strip()

            img_path = find_image(Path(info["test_img_dir"]), img_id_str, default_ext)
            if not img_path:
                skipped_test += 1
                done_items += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped_test += 1
                done_items += 1
                continue

            out_img = preprocess_image(img)
            out_name = f"{_strip_ext(img_id_str)}.png"
            save_path = out_test / out_name
            cv2.imwrite(str(save_path), out_img)

            processed_test += 1
            done_items += 1

            if done_items % 50 == 0 or done_items == total_items:
                pct = int(round(100.0 * done_items / max(total_items, 1)))
                _pp_progress(pct, done_items, total_items, "test")
                socketio.sleep(0)

            if processed_test % 300 == 0:
                _pp_log(f"✅ [{dataset_name}] Test processed: {processed_test} | skipped: {skipped_test} | {time.time()-t1:.1f}s")

        _pp_progress(100, total_items, total_items, "done")
        socketio.sleep(0)

        _pp_done(
            f"🎉 Done [{dataset_name}] | Train: {processed_train} | Val: {processed_val} (skipped {skipped_train}) | "
            f"Test: {processed_test} (skipped {skipped_test}) | Output: {out['base']}"
        )

    except Exception as e:
        _pp_err(f"❌ [{dataset_name}] Error: {e}")


@app.route("/start_preprocessing", methods=["POST"])
def start_preprocessing():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = payload.get("dataset") or "aptos2019"
    dataset_name = dataset_name.strip().lower()

    limit = payload.get("limit")
    try:
        limit_int = int(limit) if limit not in (None, "", "null") else None
    except Exception:
        limit_int = None

    if dataset_name == "all":
        def _worker_all():
            for ds in DATASET_REGISTRY.keys():
                preprocess_dataset_worker(ds, limit=limit_int)
        socketio.start_background_task(_worker_all)
        return jsonify({"status": "ok", "message": "Preprocessing started for ALL datasets..."})

    dataset_name = _safe_ds(dataset_name)
    socketio.start_background_task(preprocess_dataset_worker, dataset_name, limit_int)
    return jsonify({"status": "ok", "message": f"Preprocessing started ({dataset_name})..."})


# =========================================================
# PREPROCESS SAMPLES (Original vs Processed)
# =========================================================
@app.route("/preprocess_samples", methods=["GET"])
def preprocess_samples():
    dataset_name = _safe_ds(request.args.get("dataset") or "aptos2019")
    info = get_dataset_info(dataset_name)

    out = get_output_paths(dataset_name)
    train_dir = out["train"]
    val_dir = out["val"]
    raw_train_dir = Path(info["train_img_dir"])
    default_ext = info.get("ext", ".png")

    wanted = [0, 2, 4]
    samples = []

    def _find_one_processed_in_class(c: int) -> Optional[Path]:
        # prefer train, fallback val
        for root in (train_dir, val_dir):
            d = root / str(c)
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    return p
        return None

    for c in wanted:
        processed_path = _find_one_processed_in_class(c)
        if not processed_path:
            samples.append({"class": c, "original": "", "processed": "", "image_id": ""})
            continue

        image_id = processed_path.stem
        processed_b64 = ""
        original_b64 = ""

        proc_img = cv2.imread(str(processed_path))
        if proc_img is not None:
            proc_thumb = cv2.resize(proc_img, (256, 256), interpolation=cv2.INTER_AREA)
            processed_b64 = f"data:image/png;base64,{img_to_base64_png(proc_thumb)}"

        orig_path = find_image(raw_train_dir, image_id, default_ext)
        if orig_path:
            orig_img = cv2.imread(orig_path)
            if orig_img is not None:
                orig_thumb = cv2.resize(orig_img, (256, 256), interpolation=cv2.INTER_AREA)
                original_b64 = f"data:image/png;base64,{img_to_base64_png(orig_thumb)}"

        samples.append({
            "class": c,
            "original": original_b64,
            "processed": processed_b64,
            "image_id": image_id
        })

    return jsonify({"status": "ok", "dataset": dataset_name, "samples": samples})


# =========================================================
# EDA API
# =========================================================
@app.route("/api/eda/summary", methods=["POST"])
def api_eda_summary():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = _safe_ds(payload.get("dataset") or "aptos2019")

    info = get_dataset_info(dataset_name)
    train_csv = Path(info["train_csv"])
    if not train_csv.exists():
        return _json_error(f"Train CSV not found: {train_csv}", 400)

    df = pd.read_csv(train_csv)
    id_col = info["id_col"]
    label_col = info["label_col"]
    num_classes = int(info.get("num_classes", 5))

    if id_col not in df.columns or label_col not in df.columns:
        return _json_error(
            f"Missing columns. Need id='{id_col}', label='{label_col}'. Found: {list(df.columns)}",
            400
        )

    num_rows = int(len(df))
    missing_labels = int(df[label_col].isna().sum())

    labels_num = pd.to_numeric(df[label_col], errors="coerce")
    invalid_labels = int(labels_num.isna().sum() - missing_labels)

    valid_mask = labels_num.notna() & (labels_num >= 0) & (labels_num < num_classes)
    df_valid = df.loc[valid_mask].copy()
    num_valid_rows = int(len(df_valid))

    duplicate_ids = int(df_valid[id_col].duplicated().sum())

    labels_clean = labels_num.loc[valid_mask].astype(int)
    counts_series = labels_clean.value_counts().sort_index()

    labels = [int(x) for x in counts_series.index.tolist()]
    values = [int(x) for x in counts_series.values.tolist()]

    label_percent = {}
    if num_valid_rows > 0:
        for k, v in zip(labels, values):
            label_percent[str(k)] = float(v / num_valid_rows)

    non_zero = [v for v in values if v > 0]
    if len(non_zero) >= 2:
        imbalance_ratio = float(max(non_zero) / min(non_zero))
    elif len(non_zero) == 1:
        imbalance_ratio = float("inf")
    else:
        imbalance_ratio = None

    majority_baseline_acc = float(max(values) / num_valid_rows) if (num_valid_rows > 0 and values) else None

    fig = plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title(f"Label Distribution - {dataset_name}")
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    chart_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "status": "ok",
        "dataset": dataset_name,
        "num_rows": num_rows,
        "num_valid_rows": num_valid_rows,
        "missing_labels": missing_labels,
        "invalid_labels": invalid_labels,
        "duplicate_ids": duplicate_ids,
        "majority_baseline_accuracy": majority_baseline_acc,
        "imbalance_ratio": imbalance_ratio,
        "label_counts": {str(k): int(v) for k, v in zip(labels, values)},
        "label_percent": label_percent,
        "chart_b64": chart_b64,
    })


# =========================================================
# TRAINING
# =========================================================
@app.route("/start_training", methods=["POST"])
def start_training():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = _safe_ds(payload.get("dataset") or "aptos2019")

    out = get_output_paths(dataset_name)
    out_train = out["train"]
    out_val = out["val"]

    # ensure folders exist for both datasets
    num_classes = int(get_dataset_info(dataset_name).get("num_classes", 5))
    ensure_class_folders(out_train, num_classes=num_classes)
    ensure_class_folders(out_val, num_classes=num_classes)

    def _setdefault(k, v):
        if k not in payload or payload[k] in (None, "", "null"):
            payload[k] = v

    _setdefault("dataset", dataset_name)

    # ✅ IMPORTANT: pass processed_dir as TRAIN folder.
    # training_cnn_v2.make_datasets will detect sibling VAL folder.
    _setdefault("processed_dir", str(out_train))

    _setdefault("epochs", 20)
    _setdefault("batch_size", 16)
    _setdefault("img_size", 224)
    _setdefault("warmup_epochs", 2)
    _setdefault("lr", 1e-4)
    _setdefault("fine_tune_lr", 1e-5)
    _setdefault("dropout", 0.3)
    _setdefault("cache_dataset", False)
    _setdefault("seed", 123)

    socketio.start_background_task(run_cnn_training, socketio, payload)
    return jsonify({"status": "ok", "message": f"Training started ({dataset_name})..."})

@app.route("/api/train-three-cnn", methods=["POST"])
def api_train_three_cnn():
    return start_training()


# =========================================================
# RESULTS API
# =========================================================
_ALLOWED_RESULT_IMAGES = {"curves_accuracy.png", "curves_loss.png", "confusion_matrix.png"}

@app.route("/api/results/list", methods=["GET"])
def api_results_list():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    runs_dir = STORAGE_ROOT / dataset / "runs"
    if not runs_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    runs = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": runs})

@app.route("/api/results/get", methods=["GET"])
def api_results_get():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = (request.args.get("run_id") or "").strip()

    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        return _json_error("Invalid run_id", 400)

    run_dir = STORAGE_ROOT / dataset / "runs" / run_id
    if not run_dir.exists():
        return _json_error(f"Run not found: {run_id}", 404)

    metrics_path = run_dir / "metrics.json"
    eval_path = run_dir / "evaluation.json"
    report_path = run_dir / "report.txt"

    metrics = {}
    evaluation = {}
    report = ""

    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {"_error": "Failed to read metrics.json"}

    try:
        if eval_path.exists():
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        evaluation = {"_error": "Failed to read evaluation.json"}

    try:
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")
    except Exception:
        report = "Failed to read report.txt"

    images = {}
    for name in _ALLOWED_RESULT_IMAGES:
        p = run_dir / name
        if p.exists():
            images[name] = f"/api/results/image?dataset={dataset}&run_id={run_id}&name={name}"

    return jsonify({
        "status": "ok",
        "dataset": dataset,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "evaluation": evaluation,
        "report": report,
        "images": images,
    })

@app.route("/api/results/image", methods=["GET"])
def api_results_image():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = (request.args.get("run_id") or "").strip()
    name = (request.args.get("name") or "").strip()

    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        abort(400)
    if name not in _ALLOWED_RESULT_IMAGES:
        abort(400)

    path = STORAGE_ROOT / dataset / "runs" / run_id / name
    if not path.exists():
        abort(404)

    return send_file(path)

def _extract_model_name_from_run(run_dir: Path) -> str:
    for p in (run_dir / "metrics.json", run_dir / "evaluation.json"):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for key in ("model", "model_name", "backbone", "backbone_name"):
                val = data.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass
    return "unknown"

def _extract_created_at_from_run(run_dir: Path) -> str:
    for p in (run_dir / "metrics.json", run_dir / "evaluation.json"):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            val = data.get("created_at")
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:
            pass
    return ""

@app.route("/api/results/list_detailed", methods=["GET"])
def api_results_list_detailed():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    runs_dir = STORAGE_ROOT / dataset / "runs"
    if not runs_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    items = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        items.append({
            "run_id": d.name,
            "model_name": _extract_model_name_from_run(d),
            "created_at": _extract_created_at_from_run(d),
        })

    items.sort(key=lambda x: (runs_dir / x["run_id"]).stat().st_mtime, reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": items})


# =========================================================
# ENSEMBLE
# =========================================================
@app.route("/start_ensemble", methods=["POST"])
def start_ensemble():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}

    payload.setdefault("dataset", "aptos2019")
    payload["dataset"] = _safe_ds(payload.get("dataset"))

    # ✅ ensure processed_dir is always set consistently (train folder)
    out = get_output_paths(payload["dataset"])
    payload.setdefault("processed_dir", str(out["train"]))

    payload.setdefault("method", "softvote")
    payload.setdefault("batch_size", 16)
    payload.setdefault("img_size", 224)
    payload.setdefault("seed", 123)
    payload.setdefault("cache_dataset", False)

    socketio.start_background_task(run_ensemble, socketio, payload)
    return jsonify({"status": "ok", "message": "Ensemble started..."})


_ALLOWED_ENSEMBLE_IMAGES = {"confusion_matrix.png"}

@app.route("/api/ensemble/list", methods=["GET"])
def api_ensemble_list():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    ens_dir = STORAGE_ROOT / dataset / "ensemble_runs"
    if not ens_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    runs = sorted([p.name for p in ens_dir.iterdir() if p.is_dir()], reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": runs})

@app.route("/api/ensemble/get", methods=["GET"])
def api_ensemble_get():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = (request.args.get("run_id") or "").strip()

    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        return _json_error("Invalid run_id", 400)

    run_dir = STORAGE_ROOT / dataset / "ensemble_runs" / run_id
    if not run_dir.exists():
        return _json_error(f"Ensemble run not found: {run_id}", 404)

    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.txt"

    metrics = {}
    report = ""

    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {"_error": "Failed to read metrics.json"}

    try:
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")
    except Exception:
        report = "Failed to read report.txt"

    images = {}
    p = run_dir / "confusion_matrix.png"
    if p.exists():
        images["confusion_matrix.png"] = f"/api/ensemble/image?dataset={dataset}&run_id={run_id}&name=confusion_matrix.png"

    return jsonify({
        "status": "ok",
        "dataset": dataset,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metrics": metrics,
        "report": report,
        "images": images,
    })

@app.route("/api/ensemble/image", methods=["GET"])
def api_ensemble_image():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = (request.args.get("run_id") or "").strip()
    name = (request.args.get("name") or "").strip()

    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        abort(400)
    if name not in _ALLOWED_ENSEMBLE_IMAGES:
        abort(400)

    path = STORAGE_ROOT / dataset / "ensemble_runs" / run_id / name
    if not path.exists():
        abort(404)

    return send_file(path)


# =========================================================
# PREDICT (3 CNN + simple soft-vote ensemble)
# =========================================================
def _normalize_backbone_name(name: str) -> str:
    if not name:
        return ""
    n = str(name).strip().lower().replace("-", "").replace("_", "")
    alias = {
        "densenet": "DenseNet121",
        "densenet121": "DenseNet121",
        "inceptionresnet": "InceptionResNetV2",
        "inceptionresnetv2": "InceptionResNetV2",
        "efficientnet": "EfficientNetV2B0",
        "efficientnetv2b0": "EfficientNetV2B0",
    }
    return alias.get(n, str(name).strip())

def _find_latest_run_for_backbone(dataset: str, backbone: str) -> Dict[str, Any]:
    runs_dir = STORAGE_ROOT / dataset / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    backbone = _normalize_backbone_name(backbone)

    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        eval_path = d / "evaluation.json"
        if not eval_path.exists():
            continue
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        m = _normalize_backbone_name(data.get("model") or data.get("backbone") or data.get("model_name") or "")
        if m != backbone:
            continue

        candidates.append((d.stat().st_mtime, d, data))

    if not candidates:
        raise FileNotFoundError(f"No run found for backbone={backbone} in {runs_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, run_dir, data = candidates[0]

    class_names = data.get("class_names")
    if not isinstance(class_names, list) or len(class_names) == 0:
        num_classes = int(data.get("num_classes") or 5)
        class_names = [str(i) for i in range(num_classes)]
    else:
        num_classes = len(class_names)

    image_size = data.get("image_size") or [224, 224]
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_size = (int(image_size[0]), int(image_size[1]))
    else:
        image_size = (224, 224)

    weights_path = data.get("best_weights_path") or str(run_dir / "best.weights.h5")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"best_weights_path not found: {weights_path}")

    dropout = data.get("dropout")
    try:
        dropout = float(dropout) if dropout is not None else 0.3
    except Exception:
        dropout = 0.3

    return {
        "backbone": backbone,
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "image_size": image_size,
        "dropout": float(dropout),
    }

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Upload field: image
    Returns JSON always.
    """
    try:
        dataset = _safe_ds(request.form.get("dataset") or "aptos2019")

        f = request.files.get("image")
        if not f or not f.filename:
            return _json_error("No file uploaded. Field name must be 'image'.", 400)

        raw = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return _json_error("Failed to decode image. Upload a valid jpg/png.", 400)

        # preprocess once (224x224 base)
        proc_bgr_224 = preprocess_image(img_bgr)

        backbones = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]
        models_out = []
        probs_list = []

        for b in backbones:
            try:
                sig = _find_latest_run_for_backbone(dataset, b)

                # match training image_size
                target_w, target_h = sig["image_size"][0], sig["image_size"][1]
                proc_for_model = cv2.resize(proc_bgr_224, (target_w, target_h), interpolation=cv2.INTER_AREA)

                proc_rgb = cv2.cvtColor(proc_for_model, cv2.COLOR_BGR2RGB)
                x = np.expand_dims(proc_rgb.astype(np.float32), axis=0)  # (1,H,W,3)

                # match dropout to avoid mismatch
                model = tcnn.build_model(
                    backbone_name=sig["backbone"],
                    image_size=sig["image_size"],
                    num_classes=sig["num_classes"],
                    dropout=sig["dropout"],
                    seed=123,
                )
                model.load_weights(sig["weights_path"])

                pred = model.predict(x, verbose=0)
                probs = pred[0].astype(float).tolist()
                pred_idx = int(np.argmax(probs))
                pred_label = _label_from_index(pred_idx)

                probs_list.append(probs)

                models_out.append({
                    "model": sig["backbone"],
                    "status": "ok",
                    "run_dir": sig["run_dir"],
                    "weights_path": sig["weights_path"],
                    "num_classes": sig["num_classes"],
                    "class_names": sig["class_names"],
                    "pred_class_index": pred_idx,
                    "pred_label": pred_label,
                    "probs": probs,
                })

            except Exception as ex:
                models_out.append({
                    "model": b,
                    "status": "error",
                    "message": str(ex),
                })

        ensemble = None
        if probs_list:
            avg = np.mean(np.array(probs_list, dtype=np.float32), axis=0).tolist()
            ens_idx = int(np.argmax(avg))
            ens_label = _label_from_index(ens_idx)

            ensemble = {
                "method": "softvote_mean",
                "pred_class_index": ens_idx,
                "pred_label": ens_label,
                "probs": avg,
            }

        preview_b64 = "data:image/png;base64," + img_to_base64_png(
            cv2.resize(proc_bgr_224, (256, 256), interpolation=cv2.INTER_AREA)
        )

        return jsonify({
            "status": "ok",
            "dataset": dataset,
            "preprocessed_preview_b64": preview_b64,
            "models": models_out,
            "ensemble": ensemble,
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction crashed: {e}",
            "trace": traceback.format_exc()[:2000],
        }), 500


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    print(f"📦 STORAGE_ROOT = {STORAGE_ROOT}")
    print(f"📦 Datasets in registry: {list(DATASET_REGISTRY.keys())}")
    socketio.run(app, host="127.0.0.1", port=5000, debug=True, use_reloader=False)
