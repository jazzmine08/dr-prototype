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
    """
    p = Path(processed_dir).resolve()

    # processed_final/train -> parents[1] is dataset folder
    try:
        dataset_base = p.parents[1]
    except Exception:
        dataset_base = p.parent

    runs_base = dataset_base / "runs"
    _ensure_dir(runs_base)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_backbone = _safe_name(backbone_name)
    _ = _safe_name(dataset_name) or "unknown"

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
    # 2) Curves (robust to metric key names)
    # -----------------------
    def _plot_curve_multi(train_keys, val_keys, title: str, out_name: str):
        k_train = next((k for k in train_keys if k in hist), None)
        k_val = next((k for k in val_keys if k in hist), None)
        if (k_train is None) and (k_val is None):
            return

        fig = plt.figure()
        if k_train is not None:
            plt.plot(hist[k_train])
        if k_val is not None:
            plt.plot(hist[k_val])

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)

        labels = []
        if k_train is not None:
            labels.append("train")
        if k_val is not None:
            labels.append("val")
        if labels:
            plt.legend(labels, loc="best")

        plt.tight_layout()
        plt.savefig(run_dir / out_name, dpi=150)
        plt.close(fig)

    # accuracy sometimes saved as categorical_accuracy
    _plot_curve_multi(
        train_keys=("accuracy", "categorical_accuracy"),
        val_keys=("val_accuracy", "val_categorical_accuracy"),
        title="Accuracy",
        out_name="curves_accuracy.png",
    )
    _plot_curve_multi(
        train_keys=("loss",),
        val_keys=("val_loss",),
        title="Loss",
        out_name="curves_loss.png",
    )

    # -----------------------
    # 3) Predictions on val_ds
    # -----------------------
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for x, y in val_ds:
        pred = model.predict(x, verbose=0)
        pred_cls = np.argmax(pred, axis=1).astype(int).tolist()

        y_np = y.numpy() if hasattr(y, "numpy") else np.array(y)
        if _is_one_hot(y_np):
            true_cls = np.argmax(y_np, axis=1).astype(int).tolist()
        else:
            true_cls = y_np.astype(int).tolist()

        y_true_all.extend(true_cls)
        y_pred_all.extend(pred_cls)

    # debug label distribution (optional)
    try:
        print("DEBUG y_true counts:", np.bincount(np.array(y_true_all), minlength=5).tolist())
    except Exception as e:
        print("DEBUG y_true counts: failed:", repr(e))

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
        # -----------------------
    # Confusion matrix plot (with numbers)
    # -----------------------
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # write numbers in each cell
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(int(cm[i, j])),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10
            )

    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=170)
    plt.close(fig)

    # -----------------------
    # 4) metrics.json summary
    # -----------------------
    best_val_acc = None
    best_val_loss = None

    # best_val_accuracy might be stored under val_accuracy OR val_categorical_accuracy
    val_acc_key = "val_accuracy" if "val_accuracy" in hist else ("val_categorical_accuracy" if "val_categorical_accuracy" in hist else None)
    if val_acc_key and len(hist.get(val_acc_key, [])) > 0:
        best_val_acc = max(hist[val_acc_key])

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

        "extra": meta,
    }

    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
