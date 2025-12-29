# app.py
# Prototype: Preprocessing + EDA + Training (3 CNN) + Ensemble + Results + Predict
# ✅ ONE Flask app, ONE SocketIO
# ✅ All /api/* endpoints return JSON (even on errors)
# ✅ Admin-only pages protected
# ✅ Dataset registry + processed output paths consistent:
#    X:\dr_prototype\processed\<dataset>\processed_final\{train,val,test}\<class>\*.png
#    X:\dr_prototype\processed\<dataset>\runs\...
#    X:\dr_prototype\processed\<dataset>\ensemble_runs\...

from __future__ import annotations

import os
import json
import base64
import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from functools import wraps

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import (
    Flask, render_template, jsonify, request, send_file, abort,
    session, redirect, url_for
)
from flask_socketio import SocketIO

# Training modules
from training.trainer_three_cnn import run_cnn_training
from training.ensemble_runner import run_ensemble
import training.training_cnn_v2 as tcnn  # for custom layers + robust load

# Preprocessing module
from preprocessing import preprocess_dataset_worker, preprocess_image


# =========================================================
# CONFIGURATION & PATHS
# =========================================================
ROOT = Path(r"X:\dr_prototype")
RAW_DATASETS_ROOT = ROOT / "dataset"
STORAGE_ROOT = ROOT / "processed"

ROOT.mkdir(parents=True, exist_ok=True)
RAW_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = (Path(__file__).resolve().parent / "templates")
STATIC_DIR = (Path(__file__).resolve().parent / "static")

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = "ABC-123-SECRET-KEY-FIXED"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# =========================================================
# AUTH USERS
# =========================================================
USERS = {
    "admin": {"password": "ABC123", "role": "admin"},
    "doctor": {"password": "COBA123", "role": "user"},
}


# =========================================================
# SPLIT SETTINGS (used by preprocessing worker)
# =========================================================
VAL_RATIO = 0.20     # 80/20 split
SPLIT_SEED = 123     # deterministic split seed


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


# =========================================================
# JSON HELPERS
# =========================================================
def _json_error(message: str, http: int = 500, **extra):
    payload = {"status": "error", "message": message}
    payload.update(extra)
    return jsonify(payload), http


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
    Outputs per dataset:
      <STORAGE_ROOT>/<dataset>/processed_final/train/<0..4>/*.png
      <STORAGE_ROOT>/<dataset>/processed_final/val/<0..4>/*.png
      <STORAGE_ROOT>/<dataset>/processed_final/test/<0..4>/*.png  (if your worker creates it)
    """
    base = get_dataset_info(dataset_name)["processed_base"] / "processed_final"
    return {
        "base": base,
        "train": base / "train",
        "val": base / "val",
        "test": base / "test",
    }


def ensure_class_folders(root_dir: Path, num_classes: int = 5):
    root_dir.mkdir(parents=True, exist_ok=True)
    for c in range(num_classes):
        (root_dir / str(c)).mkdir(parents=True, exist_ok=True)


def img_to_base64_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _is_api_request() -> bool:
    return request.path.startswith("/api/")


def _safe_id(name: str) -> str:
    """Basic path traversal guard for run ids."""
    s = (name or "").strip()
    if not s or any(x in s for x in ("..", "/", "\\", ":", "\0")):
        return ""
    return s


# =========================================================
# ACCESS CONTROL
# =========================================================
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            if _is_api_request():
                return _json_error("Unauthorized. Please login.", 401)
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = session.get("user")
        if not user or user.get("role") != "admin":
            if _is_api_request():
                return _json_error("Admins only.", 403)
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


# =========================================================
# GLOBAL ERROR HANDLERS (API returns JSON)
# =========================================================
@app.errorhandler(404)
def _err_404(e):
    if _is_api_request():
        return _json_error("Not Found", 404, path=request.path)
    return e


@app.errorhandler(500)
def _err_500(e):
    if _is_api_request():
        return _json_error("Internal Server Error", 500, path=request.path)
    return e


# =========================================================
# AUTH ROUTES
# =========================================================
@app.route("/login", methods=["POST"])
def login():
    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()

    user = USERS.get(username)
    if user and user["password"] == password:
        session["user"] = {"username": username, "role": user["role"]}
        return redirect(url_for("index"))

    return "<h1>Invalid credentials</h1><a href='/'>Try again</a>", 401


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))


# =========================================================
# PAGE ROUTES
# =========================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/doctor_predict")
@login_required
def doctor_predict_page():
    if session.get("user", {}).get("role") != "user":
        return redirect(url_for("index"))
    return render_template("doctor_predict.html")


@app.route("/predict")
@admin_required
def predict_page_admin():
    return render_template("predict.html")


@app.route("/preprocessing")
@admin_required
def preprocessing_page():
    return render_template("preprocessing.html")


@app.route("/eda")
@admin_required
def eda_page():
    return render_template("eda.html")


@app.route("/training")
@admin_required
def training_page():
    return render_template("training.html")


@app.route("/ensemble")
@admin_required
def ensemble_page():
    return render_template("ensemble.html")


@app.route("/results")
@admin_required
def results_page():
    return render_template("results.html")


# =========================================================
# API: DATASETS (dropdown)
# =========================================================
@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    """
    Some frontends expect {datasets:[...]}.
    Others expect {data:[...]}.
    We return BOTH for compatibility.
    """
    ds = list(DATASET_REGISTRY.keys())
    return jsonify({"status": "ok", "datasets": ds, "data": ds})


# =========================================================
# PREPROCESSING (delegated to preprocessing.py)
# =========================================================
@app.route("/api/start_preprocessing", methods=["POST"])
@admin_required
def api_start_preprocessing():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = _safe_ds((payload.get("dataset") or "aptos2019").strip().lower())

    limit = payload.get("limit")
    try:
        limit_int = int(limit) if limit not in (None, "", "null") else None
    except Exception:
        limit_int = None

    def _run_one(ds: str):
        preprocess_dataset_worker(
            ds,
            get_dataset_info=get_dataset_info,
            get_output_paths=get_output_paths,
            val_ratio=VAL_RATIO,
            split_seed=SPLIT_SEED,
            socketio=socketio,
            limit=limit_int,
        )

    socketio.start_background_task(_run_one, dataset_name)
    return jsonify({"status": "ok", "message": f"Preprocessing started ({dataset_name})...", "dataset": dataset_name})


@app.route("/api/preprocess_samples", methods=["GET"])
@admin_required
def api_preprocess_samples():
    dataset_name = _safe_ds(request.args.get("dataset") or "aptos2019")
    info = get_dataset_info(dataset_name)

    out = get_output_paths(dataset_name)
    train_dir = out["train"]
    val_dir = out["val"]
    raw_train_dir = Path(info["train_img_dir"])
    default_ext = info.get("ext", ".png")

    wanted = [0, 2, 4]
    samples = []

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

    def _find_one_processed_in_class(c: int) -> Optional[Path]:
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
            processed_b64 = "data:image/png;base64," + img_to_base64_png(proc_thumb)

        orig_path = find_image(raw_train_dir, image_id, default_ext)
        if orig_path:
            orig_img = cv2.imread(orig_path)
            if orig_img is not None:
                orig_thumb = cv2.resize(orig_img, (256, 256), interpolation=cv2.INTER_AREA)
                original_b64 = "data:image/png;base64," + img_to_base64_png(orig_thumb)

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
@admin_required
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
@app.route("/api/start_training", methods=["POST"])
@admin_required
def api_start_training():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}
    dataset_name = _safe_ds(payload.get("dataset") or "aptos2019")

    out = get_output_paths(dataset_name)
    out_train = out["train"]
    out_val = out["val"]

    num_classes = int(get_dataset_info(dataset_name).get("num_classes", 5))
    ensure_class_folders(out_train, num_classes=num_classes)
    ensure_class_folders(out_val, num_classes=num_classes)

    def _setdefault(k, v):
        if k not in payload or payload[k] in (None, "", "null"):
            payload[k] = v

    _setdefault("dataset", dataset_name)

    # ✅ training runner expects TRAIN folder; val is sibling under processed_final
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

    _setdefault("balanced_sampling", True)
    _setdefault("use_augmentation", True)
    _setdefault("aug_rot_deg", 30.0)
    _setdefault("aug_zoom", 0.10)
    _setdefault("aug_shift", 0.05)
    _setdefault("aug_contrast", 0.15)

    # Compatibility: some code expects image_size field
    try:
        s = int(payload.get("img_size") or 224)
    except Exception:
        s = 224
    _setdefault("image_size", [s, s])

    socketio.start_background_task(run_cnn_training, socketio, payload)
    return jsonify({"status": "ok", "message": f"Training started ({dataset_name})...", "dataset": dataset_name})


@app.route("/api/train-three-cnn", methods=["POST"])
@admin_required
def api_train_three_cnn():
    return api_start_training()


# =========================================================
# RESULTS API
# =========================================================
_ALLOWED_RESULT_IMAGES = {"curves_accuracy.png", "curves_loss.png", "confusion_matrix.png"}


@app.route("/api/results/list", methods=["GET"])
@admin_required
def api_results_list():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    runs_dir = STORAGE_ROOT / dataset / "runs"
    if not runs_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": [], "items": []})

    runs = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True)

    items = []
    for rid in runs[:200]:
        run_dir = runs_dir / rid
        rec_path = run_dir / "run_record.json"
        met_path = run_dir / "metrics.json"

        rec = {}
        met = {}
        if rec_path.exists():
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            except Exception:
                rec = {}
        if met_path.exists():
            try:
                met = json.loads(met_path.read_text(encoding="utf-8"))
            except Exception:
                met = {}

        best_acc = rec.get("best_val_accuracy")
        qwk = rec.get("kappa_qwk")
        model = rec.get("model") or rec.get("backbone") or ""

        if best_acc is None:
            best_acc = (met.get("summary") or {}).get("best_val_accuracy") or met.get("best_val_accuracy")
        if qwk is None:
            qwk = (met.get("summary") or {}).get("kappa_qwk") or met.get("kappa_qwk")
        if not model:
            model = (met.get("summary") or {}).get("model") or (met.get("summary") or {}).get("backbone") or met.get("model") or met.get("backbone") or ""

        items.append({
            "run_id": rid,
            "model": model,
            "best_val_accuracy": best_acc,
            "kappa_qwk": qwk,
        })

    return jsonify({"status": "ok", "dataset": dataset, "runs": runs, "items": items})


@app.route("/api/results/get", methods=["GET"])
@admin_required
def api_results_get():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = _safe_id(request.args.get("run_id") or "")
    if not run_id:
        return _json_error("Invalid run_id", 400)

    run_dir = STORAGE_ROOT / dataset / "runs" / run_id
    if not run_dir.exists():
        return _json_error(f"Run not found: {run_id}", 404)

    metrics_path = run_dir / "metrics.json"
    eval_path = run_dir / "evaluation.json"
    report_path = run_dir / "report.txt"
    record_path = run_dir / "run_record.json"

    metrics = {}
    evaluation = {}
    report = ""
    run_record = {}

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

    try:
        if record_path.exists():
            run_record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception:
        run_record = {"_error": "Failed to read run_record.json"}

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
        "run_record": run_record,
        "images": images,
    })


@app.route("/api/results/image", methods=["GET"])
@admin_required
def api_results_image():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = _safe_id(request.args.get("run_id") or "")
    name = (request.args.get("name") or "").strip()

    if not run_id:
        abort(400)
    if name not in _ALLOWED_RESULT_IMAGES:
        abort(400)

    path = STORAGE_ROOT / dataset / "runs" / run_id / name
    if not path.exists():
        abort(404)

    return send_file(path)


# =========================================================
# ENSEMBLE
# =========================================================
@app.route("/api/start_ensemble", methods=["POST"])
@admin_required
def api_start_ensemble():
    payload = request.get_json(silent=True) or request.form.to_dict() or {}

    dataset = _safe_ds(payload.get("dataset") or "aptos2019")
    payload["dataset"] = dataset

    out = get_output_paths(dataset)
    payload.setdefault("processed_dir", str(out["train"]))

    payload.setdefault("method", "softvote")
    payload.setdefault("weighting", "by_qwk")
    payload.setdefault("batch_size", 16)
    payload.setdefault("img_size", 224)
    payload.setdefault("seed", 123)
    payload.setdefault("cache_dataset", False)
    payload.setdefault("eval_split", "val")  # "val" or "test"

    run_ids = payload.get("run_ids")
    if not (isinstance(run_ids, list) and len(run_ids) == 3):
        r1 = (payload.get("run1") or "").strip()
        r2 = (payload.get("run2") or "").strip()
        r3 = (payload.get("run3") or "").strip()
        if r1 and r2 and r3:
            payload["run_ids"] = [r1, r2, r3]

    socketio.start_background_task(run_ensemble, socketio, payload)
    return jsonify({"status": "ok", "message": "Ensemble started...", "dataset": dataset})


_ALLOWED_ENSEMBLE_IMAGES = {"confusion_matrix.png"}


@app.route("/api/ensemble/list", methods=["GET"])
@admin_required
def api_ensemble_list():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    ens_dir = STORAGE_ROOT / dataset / "ensemble_runs"
    if not ens_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    runs = sorted([p.name for p in ens_dir.iterdir() if p.is_dir()], reverse=True)
    return jsonify({"status": "ok", "dataset": dataset, "runs": runs})


@app.route("/api/ensemble/get", methods=["GET"])
@admin_required
def api_ensemble_get():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = _safe_id(request.args.get("run_id") or "")
    if not run_id:
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
@admin_required
def api_ensemble_image():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    run_id = _safe_id(request.args.get("run_id") or "")
    name = (request.args.get("name") or "").strip()

    if not run_id:
        abort(400)
    if name not in _ALLOWED_ENSEMBLE_IMAGES:
        abort(400)

    path = STORAGE_ROOT / dataset / "ensemble_runs" / run_id / name
    if not path.exists():
        abort(404)

    return send_file(path)


# =========================================================
# OPTIONAL: Dedicated endpoint for Run dropdown options
# =========================================================
@app.route("/api/runs/options", methods=["GET"])
@admin_required
def api_runs_options():
    dataset = _safe_ds(request.args.get("dataset") or "aptos2019")
    runs_dir = STORAGE_ROOT / dataset / "runs"
    if not runs_dir.exists():
        return jsonify({"status": "ok", "dataset": dataset, "runs": []})

    out = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
        rid = run_dir.name
        rec_path = run_dir / "run_record.json"
        met_path = run_dir / "metrics.json"

        rec = {}
        met = {}
        if rec_path.exists():
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
            except Exception:
                rec = {}
        if met_path.exists():
            try:
                met = json.loads(met_path.read_text(encoding="utf-8"))
            except Exception:
                met = {}

        best_acc = rec.get("best_val_accuracy")
        qwk = rec.get("kappa_qwk")
        model = rec.get("model") or rec.get("backbone") or ""

        if best_acc is None:
            best_acc = (met.get("summary") or {}).get("best_val_accuracy") or met.get("best_val_accuracy")
        if qwk is None:
            qwk = (met.get("summary") or {}).get("kappa_qwk") or met.get("kappa_qwk")
        if not model:
            model = (met.get("summary") or {}).get("model") or (met.get("summary") or {}).get("backbone") or met.get("model") or met.get("backbone") or ""

        out.append({
            "run_id": rid,
            "model": model,
            "best_val_accuracy": best_acc,
            "kappa_qwk": qwk,
        })

    return jsonify({"status": "ok", "dataset": dataset, "runs": out})


# =========================================================
# PREDICT HELPERS (REAL MODEL INFERENCE)
# =========================================================
BACKBONES = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]

# simple in-process cache so repeated predictions are fast
_MODEL_CACHE: Dict[str, Any] = {}  # path -> keras.Model
_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}  # run_dir -> metrics dict


def _label_from_index(idx: int) -> str:
    mapping = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR",
    }
    return mapping.get(int(idx), f"Class {idx}")


def _read_json_safe(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_metrics_for_run(run_dir: Path) -> Dict[str, Any]:
    k = str(run_dir)
    if k in _METRICS_CACHE:
        return _METRICS_CACHE[k]
    m = {}
    mp = run_dir / "metrics.json"
    rp = run_dir / "run_record.json"
    if mp.exists():
        m = _read_json_safe(mp)
    elif rp.exists():
        m = _read_json_safe(rp)
    _METRICS_CACHE[k] = m
    return m


def _get_run_model_path(run_dir: Path) -> Optional[Path]:
    # prefer full saved model
    p = run_dir / "best_model.keras"
    if p.exists():
        return p
    # fallback to weights only (not used here for prediction; prediction expects .keras)
    return None


def _load_keras_model_cached(model_path: Path):
    key = str(model_path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import tensorflow as tf
    from tensorflow import keras

    custom_objects = {}
    if hasattr(tcnn, "BackbonePreprocess"):
        custom_objects["BackbonePreprocess"] = tcnn.BackbonePreprocess
    if hasattr(tcnn, "RandomBrightness"):
        custom_objects["RandomBrightness"] = tcnn.RandomBrightness

    model = keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)

    # warm build
    try:
        ishape = model.input_shape
        if isinstance(ishape, (list, tuple)) and len(ishape) >= 4:
            h = int(ishape[1] or 224)
            w = int(ishape[2] or 224)
            _ = model(tf.zeros((1, h, w, 3), dtype=tf.float32), training=False)
    except Exception:
        pass

    _MODEL_CACHE[key] = model
    return model


def _latest_run_for_backbone(dataset: str, backbone: str) -> Optional[Path]:
    runs_root = STORAGE_ROOT / dataset / "runs"
    if not runs_root.exists():
        return None
    # run naming: YYYY-MM-DD_HH-MM-SS_<BackboneName>
    candidates = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.endswith(f"_{backbone}"):
            # must have model file
            if (p / "best_model.keras").exists():
                candidates.append(p)
    if not candidates:
        return None
    # lexicographic works because timestamp prefix
    candidates.sort(key=lambda x: x.name, reverse=True)
    return candidates[0]


def _resolve_selected_run_dirs(dataset: str, form: Dict[str, Any]) -> Dict[str, Path]:
    """
    Returns mapping backbone -> run_dir
    Priority:
      1) explicit run_ids[] or run1/run2/run3 (admin UI can send)
      2) latest per backbone
    """
    runs_root = STORAGE_ROOT / dataset / "runs"
    out: Dict[str, Path] = {}

    run_ids = form.get("run_ids")
    if isinstance(run_ids, list) and len(run_ids) >= 1:
        # if frontend sends list of 3, we try to map by reading metrics/model field
        for rid in run_ids:
            rid_s = _safe_id(str(rid))
            if not rid_s:
                continue
            rd = runs_root / rid_s
            if not rd.exists():
                continue
            met = _load_metrics_for_run(rd)
            model_name = (
                (met.get("model") or met.get("backbone"))
                or ((met.get("summary") or {}).get("model") or (met.get("summary") or {}).get("backbone"))
                or ""
            )
            model_name = str(model_name)
            for bb in BACKBONES:
                if bb.lower() == model_name.lower():
                    out[bb] = rd
        # if still missing, fall back later

    # legacy: run1/run2/run3
    r1 = _safe_id(str(form.get("run1") or ""))
    r2 = _safe_id(str(form.get("run2") or ""))
    r3 = _safe_id(str(form.get("run3") or ""))
    for rid in [r1, r2, r3]:
        if not rid:
            continue
        rd = runs_root / rid
        if not rd.exists():
            continue
        met = _load_metrics_for_run(rd)
        model_name = (
            (met.get("model") or met.get("backbone"))
            or ((met.get("summary") or {}).get("model") or (met.get("summary") or {}).get("backbone"))
            or ""
        )
        model_name = str(model_name)
        for bb in BACKBONES:
            if bb.lower() == model_name.lower():
                out[bb] = rd

    # fill missing with latest
    for bb in BACKBONES:
        if bb not in out:
            latest = _latest_run_for_backbone(dataset, bb)
            if latest is not None:
                out[bb] = latest

    # only keep those that have model file
    out2 = {}
    for bb, rd in out.items():
        mp = _get_run_model_path(rd)
        if mp and mp.exists():
            out2[bb] = rd
    return out2


def _auto_weights_from_runs(run_dirs: Dict[str, Path], weighting: str = "equal") -> Dict[str, float]:
    """
    weighting:
      - equal
      - by_qwk
      - by_valacc
    """
    mode = (weighting or "equal").strip().lower()
    if mode not in ("equal", "by_qwk", "by_valacc"):
        mode = "equal"

    if mode == "equal":
        n = max(1, len(run_dirs))
        return {bb: 1.0 / n for bb in run_dirs.keys()}

    scores: Dict[str, float] = {}
    for bb, rd in run_dirs.items():
        met = _load_metrics_for_run(rd)
        summary = met.get("summary") or {}
        if mode == "by_qwk":
            s = summary.get("kappa_qwk", None)
            if s is None:
                s = met.get("kappa_qwk", None)
        else:
            s = summary.get("best_val_accuracy", None)
            if s is None:
                s = met.get("best_val_accuracy", None)
        try:
            s = float(s)
        except Exception:
            s = 0.0
        if s < 0:
            s = 0.0
        scores[bb] = s

    tot = float(sum(scores.values()))
    if tot <= 0:
        n = max(1, len(run_dirs))
        return {bb: 1.0 / n for bb in run_dirs.keys()}
    return {bb: float(scores[bb] / tot) for bb in scores.keys()}


def _predict_one(model, img_bgr_proc: np.ndarray) -> Tuple[int, List[float]]:
    """
    model expects RGB float32 0..255 (because BackbonePreprocess does preprocess_input).
    img_bgr_proc: preprocessed output in BGR uint8/float, any size.
    """
    import tensorflow as tf

    # model input size
    ishape = model.input_shape
    if isinstance(ishape, (list, tuple)) and len(ishape) >= 4:
        h = int(ishape[1] or 224)
        w = int(ishape[2] or 224)
    else:
        h, w = 224, 224

    img_bgr = img_bgr_proc
    if img_bgr is None:
        raise ValueError("preprocessed image is None")

    img_resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    x = img_rgb.astype(np.float32)
    x = np.expand_dims(x, axis=0)  # [1,H,W,3]

    probs = model.predict(x, verbose=0)
    probs = np.asarray(probs).reshape(-1)
    # safety normalize
    s = float(np.sum(probs))
    if s > 0:
        probs = probs / s

    pred = int(np.argmax(probs))
    return pred, [float(p) for p in probs.tolist()]


# =========================================================
# PREDICT API (ADMIN gets per-model + ensemble, USER gets ensemble only)
# =========================================================
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    """
    Upload field: image
    Optional form fields:
      - dataset
      - weighting: equal | by_qwk | by_valacc   (admin)
      - run1/run2/run3 or run_ids[]            (admin; optional; else auto latest)
    Returns JSON always.
    """
    try:
        user = session.get("user") or {}
        role = user.get("role", "user")
        is_admin = (role == "admin")

        dataset = _safe_ds(request.form.get("dataset") or "aptos2019")

        f = request.files.get("image")
        if not f or not f.filename:
            return _json_error("No file uploaded. Field name must be 'image'.", 400)

        raw = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return _json_error("Failed to decode image. Upload a valid jpg/png.", 400)

        # Preprocess once at a high-ish resolution, then each model resizes as needed
        proc_bgr = preprocess_image(img_bgr, out_size=(512, 512))

        # Preview for UI
        preview_b64 = "data:image/png;base64," + img_to_base64_png(
            cv2.resize(proc_bgr, (256, 256), interpolation=cv2.INTER_AREA)
        )

        # Resolve selected runs (admin can send; otherwise latest per backbone)
        run_dirs = _resolve_selected_run_dirs(dataset, dict(request.form))
        if len(run_dirs) == 0:
            return _json_error(
                f"No trained models found for dataset='{dataset}'. Train models first so runs/*/best_model.keras exists.",
                400,
            )

        # Load models
        models: Dict[str, Any] = {}
        used_runs: Dict[str, str] = {}
        for bb, rd in run_dirs.items():
            mp = _get_run_model_path(rd)
            if not mp or not mp.exists():
                continue
            try:
                models[bb] = _load_keras_model_cached(mp)
                used_runs[bb] = rd.name
            except Exception as e:
                return _json_error(f"Failed to load model for {bb}: {e}", 500, trace=traceback.format_exc()[:2000])

        if len(models) == 0:
            return _json_error("No usable best_model.keras found in selected runs.", 400)

        # Per-model prediction
        per_model = {}
        for bb, model in models.items():
            pred_idx, probs = _predict_one(model, proc_bgr)
            per_model[bb] = {
                "pred_class_index": int(pred_idx),
                "pred_label": _label_from_index(pred_idx),
                "probs": probs,
            }

        # Ensemble soft-vote (weights: equal/by_qwk/by_valacc)
        weighting = (request.form.get("weighting") or "equal").strip().lower()
        w_map = _auto_weights_from_runs(run_dirs={bb: run_dirs[bb] for bb in models.keys()}, weighting=weighting)

        # Weighted probability sum
        probs_sum = None
        for bb, pm in per_model.items():
            probs = np.asarray(pm["probs"], dtype=np.float32)
            wi = float(w_map.get(bb, 0.0))
            if probs_sum is None:
                probs_sum = probs * wi
            else:
                probs_sum = probs_sum + probs * wi

        if probs_sum is None:
            return _json_error("Ensemble failed (no probabilities).", 500)

        s = float(np.sum(probs_sum))
        if s > 0:
            probs_sum = probs_sum / s

        ens_idx = int(np.argmax(probs_sum))
        ensemble = {
            "method": f"softvote_{weighting}",
            "pred_class_index": ens_idx,
            "pred_label": _label_from_index(ens_idx),
            "probs": [float(x) for x in probs_sum.tolist()],
            "weights": {k: float(v) for k, v in w_map.items()},
        }

        # Base response (both roles)
        resp: Dict[str, Any] = {
            "status": "ok",
            "dataset": dataset,
            "role": role,
            "preprocessed_preview_b64": preview_b64,
            "ensemble": ensemble,
            "used_runs": used_runs,
        }

        # Admin gets extra details
        if is_admin:
            resp["per_model"] = per_model

        return jsonify(resp)

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
    print(f"📦 ROOT = {ROOT}")
    print(f"📦 RAW_DATASETS_ROOT = {RAW_DATASETS_ROOT}")
    print(f"📦 STORAGE_ROOT = {STORAGE_ROOT}")
    print(f"📦 Datasets in registry: {list(DATASET_REGISTRY.keys())}")

    socketio.run(app, host="127.0.0.1", port=5000, debug=True, use_reloader=False)
