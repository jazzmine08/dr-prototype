# ======================================
# app.py  (READY TO COPY-PASTE WHOLE FILE)
# ======================================
# Prototype: Preprocessing + EDA + Training (3 CNN) + Ensemble + Results (APTOS + DRTiD)

import os
import time
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, jsonify, request, send_file, abort
from flask_socketio import SocketIO

from training.trainer_three_cnn import run_cnn_training
from training.ensemble_runner import run_ensemble


# =========================================================
# PATHS + STORAGE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

DRIVE_ROOT = Path(r"G:\My Drive")
RAW_DATASETS_ROOT = DRIVE_ROOT / "dataset"

# Where to store standardized outputs (preprocessed images, runs, ensemble_runs, etc.)
PROCESSED_ROOT = DRIVE_ROOT / "dr_prototype" / "processed"

# Local fallback if G: isn't available
LOCAL_FALLBACK_ROOT = BASE_DIR / "dr_prototype_fallback" / "processed"

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
        # ⚠️ change these 2 if your DRTiD CSV uses different names
        "id_col": "id_code",
        "label_col": "diagnosis",
        "ext": ".png",
        "num_classes": 5,
    },
}


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASET_REGISTRY.keys())}")
    info = DATASET_REGISTRY[dataset_name].copy()
    info["processed_base"] = STORAGE_ROOT / dataset_name
    info["processed_base"].mkdir(parents=True, exist_ok=True)
    return info


def get_output_paths(dataset_name: str) -> Dict[str, Path]:
    """
    Output per dataset:
      <STORAGE_ROOT>/<dataset>/processed_final/train/<0..4>/
      <STORAGE_ROOT>/<dataset>/processed_final/test/
    """
    base = get_dataset_info(dataset_name)["processed_base"] / "processed_final"
    return {
        "base": base,
        "train": base / "train",
        "test": base / "test",
    }


def ensure_class_folders(train_dir: Path, num_classes: int = 5):
    for c in range(num_classes):
        (train_dir / str(c)).mkdir(parents=True, exist_ok=True)


# =========================================================
# FLASK + SOCKETIO
# =========================================================
app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = "secretkey"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# =========================================================
# PAGES (NAV HEADER NEEDS THESE ROUTES)
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


# =========================================================
# API: list datasets (dropdown)
# =========================================================
@app.route("/datasets", methods=["GET"])
def list_datasets():
    return jsonify({"datasets": list(DATASET_REGISTRY.keys())})


# =========================================================
# HELPERS
# =========================================================
def _strip_ext(name: str) -> str:
    base = str(name).strip()
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        if base.lower().endswith(ext):
            return base[: -len(ext)]
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


# =========================================================
# PREPROCESSING PIPELINE
# =========================================================
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        img_bgr = img_bgr[y:y + h, x:x + w]

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    img = img_bgr.astype(np.float32)
    mean = img.mean(axis=(0, 1))
    img *= mean.mean() / (mean + 1e-6)
    img_bgr = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    img_bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    g = img_bgr[:, :, 1]
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    g = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    img_bgr[:, :, 1] = np.clip(g, 0, 255).astype(np.uint8)

    return cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)


def _pp_log(msg: str):
    socketio.emit("preprocess_log", {"message": msg})


def _pp_err(msg: str):
    socketio.emit("preprocess_error", {"message": msg})


def _pp_done(msg: str):
    socketio.emit("preprocess_done", {"message": msg})


def preprocess_dataset_worker(dataset_name: str, limit: Optional[int] = None):
    try:
        info = get_dataset_info(dataset_name)
        out = get_output_paths(dataset_name)
        out_train = out["train"]
        out_test = out["test"]

        out_train.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)
        ensure_class_folders(out_train, num_classes=int(info.get("num_classes", 5)))

        # Validate raw paths
        for k in ["train_csv", "train_img_dir", "test_csv", "test_img_dir"]:
            p = Path(info[k])
            if not p.exists():
                _pp_err(f"❌ [{dataset_name}] Path not found: {p}")
                return

        _pp_log(f"🟢 Preprocessing started [{dataset_name}]")
        _pp_log(f"📦 Output base: {out['base']}")

        train_df = pd.read_csv(info["train_csv"])
        test_df = pd.read_csv(info["test_csv"])

        id_col = info["id_col"]
        label_col = info["label_col"]
        default_ext = info["ext"]

        if id_col not in train_df.columns:
            _pp_err(f"❌ [{dataset_name}] train csv missing '{id_col}'. Columns: {list(train_df.columns)}")
            return
        if label_col not in train_df.columns:
            _pp_err(f"❌ [{dataset_name}] train csv missing '{label_col}'. Columns: {list(train_df.columns)}")
            return

        # test id column fallback
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

        # TRAIN
        processed_train = 0
        skipped_train = 0
        t0 = time.time()

        for _, row in train_df.iterrows():
            img_id = row[id_col]
            label = int(row[label_col])

            img_path = find_image(Path(info["train_img_dir"]), str(img_id), default_ext)
            if not img_path:
                skipped_train += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped_train += 1
                continue

            out_img = preprocess_image(img)
            out_name = f"{_strip_ext(str(img_id))}.png"
            save_path = out_train / str(label) / out_name
            cv2.imwrite(str(save_path), out_img)

            processed_train += 1
            if processed_train % 200 == 0:
                _pp_log(f"✅ [{dataset_name}] Train processed: {processed_train} | skipped: {skipped_train} | {time.time()-t0:.1f}s")

        # TEST
        processed_test = 0
        skipped_test = 0
        t1 = time.time()

        for _, row in test_df.iterrows():
            img_id = row[test_id_col]
            img_path = find_image(Path(info["test_img_dir"]), str(img_id), default_ext)
            if not img_path:
                skipped_test += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped_test += 1
                continue

            out_img = preprocess_image(img)
            out_name = f"{_strip_ext(str(img_id))}.png"
            save_path = out_test / out_name
            cv2.imwrite(str(save_path), out_img)

            processed_test += 1
            if processed_test % 200 == 0:
                _pp_log(f"✅ [{dataset_name}] Test processed: {processed_test} | skipped: {skipped_test} | {time.time()-t1:.1f}s")

        _pp_done(
            f"🎉 Done [{dataset_name}] | Train: {processed_train} (skipped {skipped_train}) | "
            f"Test: {processed_test} (skipped {skipped_test}) | Output: {out['base']}"
        )

    except Exception as e:
        _pp_err(f"❌ [{dataset_name}] Error: {e}")


@app.route("/start_preprocessing", methods=["POST"])
def start_preprocessing():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = (payload.get("dataset") or "aptos2019").strip().lower()

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

    if dataset_name not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset_name}'"}), 400

    socketio.start_background_task(preprocess_dataset_worker, dataset_name, limit_int)
    return jsonify({"status": "ok", "message": f"Preprocessing started ({dataset_name})..."})


# =========================================================
# EDA API
# =========================================================
@app.route("/api/eda/summary", methods=["POST"])
def api_eda_summary():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = (payload.get("dataset") or "aptos2019").strip().lower()

    if dataset_name not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset_name}'"}), 400

    info = get_dataset_info(dataset_name)
    train_csv = Path(info["train_csv"])
    if not train_csv.exists():
        return jsonify({"status": "error", "message": f"Train CSV not found: {train_csv}"}), 400

    df = pd.read_csv(train_csv)
    id_col = info["id_col"]
    label_col = info["label_col"]
    if id_col not in df.columns or label_col not in df.columns:
        return jsonify({
            "status": "error",
            "message": f"Missing columns. Need id='{id_col}', label='{label_col}'. Found: {list(df.columns)}"
        }), 400

    counts = df[label_col].value_counts().sort_index()
    labels = [int(x) for x in counts.index.tolist()]
    values = [int(x) for x in counts.values.tolist()]

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

    img_dir = Path(info["train_img_dir"])
    sample_rows = df.sample(n=min(6, len(df)), random_state=42)
    samples = []
    for _, r in sample_rows.iterrows():
        img_id = r[id_col]
        label = int(r[label_col])
        p = find_image(img_dir, str(img_id), info["ext"])
        if not p:
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        thumb = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        samples.append({"id": str(img_id), "label": label, "image_b64": img_to_base64_png(thumb)})

    return jsonify({
        "status": "ok",
        "dataset": dataset_name,
        "num_rows": int(len(df)),
        "label_counts": dict(zip(map(str, labels), values)),
        "chart_b64": chart_b64,
        "samples": samples
    })


# =========================================================
# TRAINING
# =========================================================
@app.route("/start_training", methods=["POST"])
def start_training():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = (payload.get("dataset") or "aptos2019").strip().lower()

    if dataset_name not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset_name}'"}), 400

    out = get_output_paths(dataset_name)
    out_train = out["train"]
    ensure_class_folders(out_train, num_classes=int(get_dataset_info(dataset_name).get("num_classes", 5)))

    def _setdefault(k, v):
        if k not in payload or payload[k] in (None, "", "null"):
            payload[k] = v

    _setdefault("dataset", dataset_name)
    _setdefault("processed_dir", str(out_train))
    _setdefault("epochs", 20)
    _setdefault("batch_size", 16)
    _setdefault("img_size", 224)
    _setdefault("warmup_epochs", 2)
    _setdefault("lr", 1e-4)
    _setdefault("fine_tune_lr", 1e-5)
    _setdefault("dropout", 0.3)

    socketio.start_background_task(run_cnn_training, socketio, payload)
    return jsonify({"status": "ok", "message": f"Training started ({dataset_name})..." })


@app.route("/api/train-three-cnn", methods=["POST"])
def api_train_three_cnn():
    return start_training()


# =========================================================
# RESULTS API (per-dataset runs)
# =========================================================
_ALLOWED_RESULT_IMAGES = {"curves_accuracy.png", "curves_loss.png", "confusion_matrix.png"}

@app.route("/api/results/list", methods=["GET"])
def api_results_list():
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    if dataset not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset}'"}), 400

    runs_dir = STORAGE_ROOT / dataset / "runs"
    if not runs_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    runs = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": runs})


@app.route("/api/results/get", methods=["GET"])
def api_results_get():
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    run_id = (request.args.get("run_id") or "").strip()

    if dataset not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset}'"}), 400
    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        return jsonify({"status": "error", "message": "Invalid run_id"}), 400

    run_dir = STORAGE_ROOT / dataset / "runs" / run_id
    if not run_dir.exists():
        return jsonify({"status": "error", "message": f"Run not found: {run_id}"}), 404

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
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    run_id = (request.args.get("run_id") or "").strip()
    name = (request.args.get("name") or "").strip()

    if dataset not in DATASET_REGISTRY:
        abort(400)
    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        abort(400)
    if name not in _ALLOWED_RESULT_IMAGES:
        abort(400)

    path = STORAGE_ROOT / dataset / "runs" / run_id / name
    if not path.exists():
        abort(404)

    return send_file(path)

# =========================================================
# RESULTS API (detailed for ensemble dropdown labels)
#   returns: [{run_id, model_name, created_at}, ...]
# =========================================================
def _extract_model_name_from_run(run_dir: Path) -> str:
    """
    Try to read model/backbone name from metrics.json or evaluation.json.
    Works even if some keys are missing.
    """
    candidates = [run_dir / "metrics.json", run_dir / "evaluation.json"]
    for p in candidates:
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
    candidates = [run_dir / "metrics.json", run_dir / "evaluation.json"]
    for p in candidates:
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
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    if dataset not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset}'"}), 400

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

    # newest first (by folder modified time)
    items.sort(key=lambda x: (runs_dir / x["run_id"]).stat().st_mtime, reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": items})


# =========================================================
# ENSEMBLE
# =========================================================
@app.route("/start_ensemble", methods=["POST"])
def start_ensemble():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}

    payload.setdefault("method", "softvote")
    payload.setdefault("batch_size", 16)
    payload.setdefault("img_size", 224)
    payload.setdefault("seed", 123)
    payload.setdefault("cache_dataset", False)

    socketio.start_background_task(run_ensemble, socketio, payload)
    return jsonify({"status": "ok", "message": "Ensemble started..." })


_ALLOWED_ENSEMBLE_IMAGES = {"confusion_matrix.png"}

@app.route("/api/ensemble/list", methods=["GET"])
def api_ensemble_list():
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    if dataset not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset}'"}), 400

    ens_dir = STORAGE_ROOT / dataset / "ensemble_runs"
    if not ens_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    runs = sorted([p.name for p in ens_dir.iterdir() if p.is_dir()], reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": runs})


@app.route("/api/ensemble/get", methods=["GET"])
def api_ensemble_get():
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    run_id = (request.args.get("run_id") or "").strip()

    if dataset not in DATASET_REGISTRY:
        return jsonify({"status": "error", "message": f"Unknown dataset '{dataset}'"}), 400
    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        return jsonify({"status": "error", "message": "Invalid run_id"}), 400

    run_dir = STORAGE_ROOT / dataset / "ensemble_runs" / run_id
    if not run_dir.exists():
        return jsonify({"status": "error", "message": f"Ensemble run not found: {run_id}"}), 404

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
    dataset = (request.args.get("dataset") or "aptos2019").strip().lower()
    run_id = (request.args.get("run_id") or "").strip()
    name = (request.args.get("name") or "").strip()

    if dataset not in DATASET_REGISTRY:
        abort(400)
    if not run_id or any(x in run_id for x in ("..", "/", "\\", ":", "\0")):
        abort(400)
    if name not in _ALLOWED_ENSEMBLE_IMAGES:
        abort(400)

    path = STORAGE_ROOT / dataset / "ensemble_runs" / run_id / name
    if not path.exists():
        abort(404)

    return send_file(path)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    print(f"📦 STORAGE_ROOT = {STORAGE_ROOT}")
    socketio.run(app, host="127.0.0.1", port=5000, debug=True, use_reloader=False)
