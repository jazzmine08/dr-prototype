# training/results_utils.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score


# -----------------------
# helpers
# -----------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _is_one_hot(y: np.ndarray) -> bool:
    return y.ndim == 2 and y.shape[1] > 1


def _safe_name(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isalnum() or ch in ("-", "_")).strip("_")


# -----------------------
# SINGLE SOURCE OF TRUTH:
# run folder creator
# -----------------------
def make_run_dir(processed_dir: Union[str, Path], dataset_name: str, backbone_name: str) -> Path:
    """
    Creates run dir inside:
      <dataset_base>/runs/<timestamp>_<backbone>

    Expected processed_dir:
      .../<dataset>/processed_final/train
    Then dataset_base = .../<dataset>/

    Example:
      G:\\My Drive\\dr_prototype\\processed\\aptos2019\\runs\\2025-12-16_15-22-10_DenseNet50
    """
    p = Path(processed_dir).resolve()

    # processed_final/train -> parents[1] is dataset folder
    # .../<dataset>/processed_final/train
    try:
        dataset_base = p.parents[1]
    except Exception:
        # fallback: put runs next to processed_dir
        dataset_base = p.parent

    runs_base = dataset_base / "runs"
    _ensure_dir(runs_base)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_backbone = _safe_name(backbone_name)
    safe_dataset = _safe_name(dataset_name) or "unknown"

    run_dir = runs_base / f"{run_id}_{safe_backbone}"
    _ensure_dir(run_dir)
    return run_dir


# -----------------------
# Save artifacts
# -----------------------
def save_training_artifacts(
    run_dir: Union[Path, str],
    history: Union[Dict[str, List[float]], Any],
    model,
    val_ds,
    class_names: List[str],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Saves artifacts into run_dir:
      - history.csv
      - curves_accuracy.png, curves_loss.png
      - confusion_matrix.png
      - report.txt
      - metrics.json

    IMPORTANT: Does NOT create run_dir; caller creates it via make_run_dir().
    """
    run_dir = Path(run_dir)
    _ensure_dir(run_dir)

    meta = meta or {}
    dataset = str(meta.get("dataset", "unknown"))
    backbone = str(meta.get("backbone", meta.get("model", "unknown")))
    created_at = meta.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -----------------------
    # 1) History
    # -----------------------
    hist = getattr(history, "history", None)
    if hist is None and isinstance(history, dict):
        hist = history
    if hist is None:
        hist = {}

    hist_df = pd.DataFrame(hist)
    hist_path = run_dir / "history.csv"
    hist_df.to_csv(hist_path, index=False)

    # -----------------------
    # 2) Curves
    # -----------------------
    def _plot_curve(keys: Tuple[str, str], title: str, out_name: str):
        k_train, k_val = keys
        if (k_train not in hist) and (k_val not in hist):
            return
        fig = plt.figure()
        if k_train in hist:
            plt.plot(hist[k_train])
        if k_val in hist:
            plt.plot(hist[k_val])
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        labels = []
        if k_train in hist:
            labels.append("train")
        if k_val in hist:
            labels.append("val")
        if labels:
            plt.legend(labels, loc="best")
        plt.tight_layout()
        plt.savefig(run_dir / out_name, dpi=150)
        plt.close(fig)

    _plot_curve(("accuracy", "val_accuracy"), "Accuracy", "curves_accuracy.png")
    _plot_curve(("loss", "val_loss"), "Loss", "curves_loss.png")

    # -----------------------
    # 3) Predictions on val_ds
    # -----------------------
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for batch in val_ds:
        x, y = batch
        pred = model.predict(x, verbose=0)
        pred_cls = np.argmax(pred, axis=1).astype(int).tolist()

        y_np = y.numpy() if hasattr(y, "numpy") else np.array(y)
        if _is_one_hot(y_np):
            true_cls = np.argmax(y_np, axis=1).astype(int).tolist()
        else:
            true_cls = y_np.astype(int).tolist()

        y_true_all.extend(true_cls)
        y_pred_all.extend(pred_cls)

    num_classes = len(class_names)
    labels = list(range(num_classes))

    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
    kappa_qwk = cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic")

    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (run_dir / "report.txt").write_text(report, encoding="utf-8")

    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=45, ha="right")
    plt.yticks(range(num_classes), class_names)
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # -----------------------
    # 4) metrics.json summary
    # -----------------------
    best_val_acc = None
    best_val_loss = None
    if "val_accuracy" in hist and len(hist["val_accuracy"]) > 0:
        best_val_acc = max(hist["val_accuracy"])
    if "val_loss" in hist and len(hist["val_loss"]) > 0:
        best_val_loss = min(hist["val_loss"])

    summary = {
        "dataset": dataset,
        "model": backbone,
        "backbone": backbone,
        "created_at": created_at,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),

        "kappa_qwk": _safe_float(kappa_qwk),
        "best_val_accuracy": _safe_float(best_val_acc),
        "best_val_loss": _safe_float(best_val_loss),
        "num_val_samples": int(len(y_true_all)),

        "artifacts": {
            "history_csv": str(hist_path),
            "curves_accuracy.png": str(run_dir / "curves_accuracy.png"),
            "curves_loss.png": str(run_dir / "curves_loss.png"),
            "confusion_matrix.png": str(run_dir / "confusion_matrix.png"),
            "report.txt": str(run_dir / "report.txt"),
        },

        "extra": meta,  # keep all meta for traceability
    }

    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
