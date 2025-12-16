# training/ensemble_runner.py
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    cohen_kappa_score,
)

import tensorflow as tf
from tensorflow import keras

import training.training_cnn_v2 as tcnn


# ---------------------------
# helpers
# ---------------------------
def _emit(socketio, event: str, payload: dict):
    try:
        if socketio is not None:
            socketio.emit(event, payload)
    except Exception:
        # do not crash if socket emit fails
        pass


def _is_one_hot(y: np.ndarray) -> bool:
    return y.ndim == 2 and y.shape[1] > 1


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _find_model_file(run_dir: Path) -> Optional[Path]:
    """
    Prefer best_model.keras, fallback to any .keras/.h5.
    """
    p = run_dir / "best_model.keras"
    if p.exists():
        return p
    for ext in (".keras", ".h5", ".hdf5"):
        files = list(run_dir.glob(f"*{ext}"))
        if files:
            # pick newest
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return files[0]
    return None


def _infer_dataset_base_from_processed_dir(processed_dir: Path) -> Path:
    """
    processed_dir expected like:
      .../processed/<dataset>/processed_final/train
    dataset_base = .../processed/<dataset>
    """
    p = processed_dir.resolve()
    # train -> processed_final -> <dataset>
    return p.parents[1].parent  # processed_final parent is dataset_base


def _make_ensemble_run_dir(dataset_base: Path, method: str) -> Path:
    out = dataset_base / "ensemble_runs"
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out / f"{run_id}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_three_models(runs_root: Path, run_ids: List[str]) -> Tuple[List[keras.Model], List[str]]:
    model_paths = []
    for rid in run_ids:
        # basic safety against traversal
        if any(x in rid for x in ("..", "/", "\\", ":", "\0")):
            raise ValueError(f"Invalid run_id: {rid}")

        run_dir = runs_root / rid
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")

        mf = _find_model_file(run_dir)
        if not mf:
            raise FileNotFoundError(f"No model file found in: {run_dir} (expected best_model.keras or *.keras/*.h5)")
        model_paths.append(mf)

    models = []
    for mp in model_paths:
        # safe_mode=False is needed if your model has Lambda layers
        m = keras.models.load_model(str(mp), safe_mode=False, compile=False)
        models.append(m)

    return models, [str(p) for p in model_paths]


def _save_confusion_matrix_png(cm: np.ndarray, class_names: List[str], out_path: Path):
    plt.figure(figsize=(6.5, 6.0))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Ensemble Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160)
    plt.close()


# ---------------------------
# main entry
# ---------------------------
def run_ensemble(socketio, payload: Dict[str, Any]):
    """
    Payload expected (from app.py):
      dataset: "aptos2019" or "drtid"
      processed_dir: ".../processed/<dataset>/processed_final/train"
      run_ids: [run1, run2, run3]  (folder names under .../processed/<dataset>/runs/)
      method: "softvote" (default)
      weights: optional [w1,w2,w3] (default equal)
      batch_size: int (default 16)
      img_size: int (default 224)
      seed: int (default 123)
      cache_dataset: bool (default False)

    Emits:
      ensemble_log, ensemble_error, ensemble_done
    """
    t0 = time.time()
    try:
        dataset = (payload.get("dataset") or "unknown").strip().lower()
        processed_dir = Path(payload.get("processed_dir") or "").resolve()
        run_ids = payload.get("run_ids") or []
        method = (payload.get("method") or "softvote").strip().lower()

        batch_size = int(payload.get("batch_size", 16))
        img_size = int(payload.get("img_size", 224))
        seed = int(payload.get("seed", 123))
        cache_dataset = bool(payload.get("cache_dataset", False))

        weights_in = payload.get("weights")
        if weights_in and isinstance(weights_in, list) and len(weights_in) == 3:
            w = np.array([float(x) for x in weights_in], dtype=np.float32)
        else:
            w = np.ones(3, dtype=np.float32)
        if w.sum() <= 0:
            w = np.ones(3, dtype=np.float32)
        w = w / (w.sum() + 1e-8)

        if not processed_dir.exists():
            _emit(socketio, "ensemble_error", {"message": f"processed_dir not found: {processed_dir}"})
            return None
        if len(run_ids) != 3:
            _emit(socketio, "ensemble_error", {"message": "Please select EXACTLY 3 runs for ensemble."})
            return None

        _emit(socketio, "ensemble_log", {"message": f"🧩 Ensemble start | dataset={dataset} | method={method}"})
        _emit(socketio, "ensemble_log", {"message": f"📁 processed_dir: {str(processed_dir)}"})
        _emit(socketio, "ensemble_log", {"message": f"🧠 runs: {run_ids}"})
        _emit(socketio, "ensemble_log", {"message": f"⚖️ weights: {w.tolist()}"})


        dataset_base = _infer_dataset_base_from_processed_dir(processed_dir)
        runs_root = dataset_base / "runs"

        _emit(socketio, "ensemble_log", {"message": f"📦 Loading 3 models from: {str(runs_root)}"})
        models, model_paths = _load_three_models(runs_root, run_ids)
        _emit(socketio, "ensemble_log", {"message": f"✅ Loaded models: {model_paths}"})


        # build validation dataset the same way training does
        _emit(socketio, "ensemble_log", {"message": "📚 Building validation dataset (same split as training)..."})
        train_ds, val_ds, class_names, _steps = tcnn.make_datasets(
            processed_dir=str(processed_dir),
            image_size=(img_size, img_size),
            batch_size=batch_size,
            cache=cache_dataset,
            seed=seed,
            socketio=socketio,
        )
        num_classes = len(class_names)
        if num_classes <= 0:
            raise RuntimeError("No classes found in dataset folder.")

        # predict on val_ds
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        _emit(socketio, "ensemble_log", {"message": "🔮 Predicting ensemble on val set..."})
        for xb, yb in val_ds:
            y_np = yb.numpy() if hasattr(yb, "numpy") else np.array(yb)
            if _is_one_hot(y_np):
                true_cls = np.argmax(y_np, axis=1).astype(int)
            else:
                true_cls = y_np.astype(int)

            probs_sum = None
            for i, m in enumerate(models):
                probs = m.predict(xb, verbose=0)
                probs_sum = probs * w[i] if probs_sum is None else probs_sum + probs * w[i]

            pred_cls = np.argmax(probs_sum, axis=1).astype(int)

            y_true_all.extend(true_cls.tolist())
            y_pred_all.extend(pred_cls.tolist())

        # metrics
        acc = accuracy_score(y_true_all, y_pred_all)
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average="macro", zero_division=0
        )
        qwk = cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic")

        cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(num_classes)))
        report_txt = classification_report(
            y_true_all, y_pred_all,
            labels=list(range(num_classes)),
            target_names=class_names,
            digits=4,
            zero_division=0
        )

        # save artifacts
        run_dir = _make_ensemble_run_dir(dataset_base, method)
        (run_dir / "report.txt").write_text(report_txt, encoding="utf-8")
        _save_confusion_matrix_png(cm, class_names, run_dir / "confusion_matrix.png")

        summary = {
            "dataset": dataset,
            "method": method,
            "run_ids": run_ids,
            "model_paths": model_paths,
            "weights": w.tolist(),
            "num_val_samples": int(len(y_true_all)),
            "accuracy": _safe_float(acc),
            "precision_macro": _safe_float(p_macro),
            "recall_macro": _safe_float(r_macro),
            "f1_macro": _safe_float(f_macro),
            "kappa_qwk": _safe_float(qwk),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "artifacts": {
                "run_dir": str(run_dir),
                "report_txt": str(run_dir / "report.txt"),
                "confusion_matrix_png": str(run_dir / "confusion_matrix.png"),
            }
        }
        (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        elapsed = time.time() - t0
        _emit(socketio, "ensemble_log", {"message": f"✅ Ensemble done in {elapsed:.1f}s | acc={acc:.4f} | QWK={qwk:.4f}"})
        _emit(socketio, "ensemble_done", {"status": "ok", "run_dir": str(run_dir), "summary": summary})
        return summary

    except Exception as e:
        _emit(socketio, "ensemble_error", {"message": str(e), "trace": traceback.format_exc()})
        return None
