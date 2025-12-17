# ensemble_runner.py
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score

import training.training_cnn_v2 as tcnn


# ---------------------------
# helpers
# ---------------------------
def _emit(socketio, event: str, payload: dict):
    """Safe emit (won't crash if socketio is None)."""
    try:
        if socketio is not None:
            socketio.emit(event, payload)
        else:
            print(f"[{event}] {payload}")
    except Exception:
        pass


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_one_hot(y: np.ndarray) -> bool:
    if y is None:
        return False
    y = np.asarray(y)
    if y.ndim != 2:
        return False
    # typical one-hot: rows sum to 1, values 0/1 (or floats close)
    row_sum = np.sum(y, axis=1)
    return np.allclose(row_sum, 1.0, atol=1e-3)


def _safe_run_id(run_id: str) -> str:
    # prevent path traversal
    run_id = (run_id or "").strip()
    if not run_id:
        return ""
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        return ""
    return run_id


def _infer_dataset_base_from_processed_dir(processed_dir: Path) -> Path:
    """
    processed_dir example:
      G:\\My Drive\\dr_prototype\\processed\\aptos2019\\processed_final\\train

    In your codebase, run folders are created under:
      <dataset_base>/runs/<timestamp>_<backbone>
    where dataset_base is the parent of processed_final (= p.parents[1] from train dir).
    See trainer_three_cnn + results_utils usage. :contentReference[oaicite:2]{index=2}
    """
    p = Path(processed_dir).resolve()
    # train -> processed_final -> <dataset_base> (aptos2019)
    return p.parents[1]


def _resolve_weights_path(run_dir: Path, metrics: dict) -> Optional[Path]:
    """
    Prefer:
      run_dir/best.weights.h5
    Fallback:
      metrics["extra"]["best_weights_path"] if exists
    Or any *.weights.h5 inside run_dir
    """
    p1 = run_dir / "best.weights.h5"
    if p1.exists():
        return p1

    # metrics.json usually stores the path in extra.best_weights_path :contentReference[oaicite:3]{index=3}
    extra = metrics.get("extra") or {}
    bp = extra.get("best_weights_path")
    if bp:
        bp_path = Path(str(bp))
        if bp_path.exists():
            return bp_path

    # fallback glob
    for cand in run_dir.glob("*.weights.h5"):
        if cand.exists():
            return cand

    return None


def _load_run_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return _read_json(metrics_path)
    return {}


def _extract_run_meta(metrics: dict, payload: dict) -> dict:
    """
    Tries to pull image_size, dropout, seed, num_classes from metrics.json,
    otherwise fallback to payload defaults.
    """
    extra = metrics.get("extra") or {}

    # image_size stored as list in trainer meta :contentReference[oaicite:4]{index=4}
    img = extra.get("image_size")
    if isinstance(img, list) and len(img) == 2:
        image_size = (int(img[0]), int(img[1]))
    else:
        img_size = int(payload.get("img_size", 224))
        image_size = (img_size, img_size)

    num_classes = int(extra.get("num_classes") or 0)
    dropout = float(extra.get("dropout") or payload.get("dropout", 0.3))
    seed = int(extra.get("seed") or payload.get("seed", 123))

    return {
        "image_size": image_size,
        "num_classes": num_classes,
        "dropout": dropout,
        "seed": seed,
    }


def _extract_backbone(metrics: dict, run_id: str) -> str:
    # metrics.json summary uses "model" as backbone name :contentReference[oaicite:5]{index=5}
    summary = metrics.get("summary") or {}
    b = summary.get("model") or summary.get("backbone") or ""
    b = str(b).strip()
    if b:
        return b

    # fallback parse from run_id like: 2025-12-17_22-32-01_InceptionResNetV2
    parts = (run_id or "").split("_")
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return run_id


def _auto_weights(
    weighting: str,
    metrics_list: List[dict],
) -> np.ndarray:
    """
    weighting:
      - equal
      - by_qwk   (prefer this)
      - by_valacc
    """
    weighting = (weighting or "equal").strip().lower()
    if weighting not in ("equal", "by_qwk", "by_valacc"):
        weighting = "equal"

    if weighting == "equal":
        w = np.ones(len(metrics_list), dtype=np.float32)
        return w / (w.sum() + 1e-8)

    scores = []
    for m in metrics_list:
        summary = m.get("summary") or {}
        if weighting == "by_qwk":
            s = summary.get("kappa_qwk", None)
        else:
            s = summary.get("best_val_accuracy", None)

        try:
            s = float(s)
        except Exception:
            s = 0.0

        # clip negatives
        if s < 0:
            s = 0.0
        scores.append(s)

    arr = np.array(scores, dtype=np.float32)
    if arr.sum() <= 0:
        arr = np.ones(len(metrics_list), dtype=np.float32)
    return arr / (arr.sum() + 1e-8)


def _save_ensemble_artifacts(
    out_dir: Path,
    class_names: List[str],
    y_true: List[int],
    y_pred: List[int],
    meta: dict,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # report.txt
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (out_dir / "report.txt").write_text(report, encoding="utf-8")

    # confusion_matrix.png
    fig = plt.figure(figsize=(6.5, 6.0))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Ensemble Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(str(out_dir / "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    acc = float(accuracy_score(y_true, y_pred))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    metrics_payload = {
        "summary": {
            "accuracy": acc,
            "kappa_qwk": qwk,
            "num_samples": int(len(y_true)),
        },
        "extra": meta,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {"accuracy": acc, "kappa_qwk": qwk}


# ---------------------------
# main entry
# ---------------------------
def run_ensemble(socketio, payload: Dict[str, Any]):
    """
    Payload expected (from ensemble.html / app.py):
      dataset: "aptos2019" or "drtid"
      processed_dir: ".../processed/<dataset>/processed_final/train"
      run_ids: [run1, run2, run3]  (folder names under <dataset_base>/runs/)
      method: "softvote" or "hardvote"
      weights: optional [w1,w2,w3]
      weighting: optional "equal" | "by_qwk" | "by_valacc"
      batch_size: int
      img_size: int (fallback only; we will prefer metrics.json image_size)
      seed: int (fallback only)
      cache_dataset: bool

    Emits:
      ensemble_log, ensemble_error, ensemble_done
    """
    t0 = time.time()
    try:
        dataset = (payload.get("dataset") or "unknown").strip().lower()
        processed_dir = Path(payload.get("processed_dir") or "").resolve()
        run_ids_in = payload.get("run_ids") or []
        method = (payload.get("method") or "softvote").strip().lower()
        weighting_mode = (payload.get("weighting") or "by_qwk").strip().lower()

        batch_size = int(payload.get("batch_size", 16))
        cache_dataset = bool(payload.get("cache_dataset", False))

        # manual weights (optional)
        weights_in = payload.get("weights")
        manual_weights = None
        if weights_in and isinstance(weights_in, list) and len(weights_in) == 3:
            try:
                w = np.array([float(x) for x in weights_in], dtype=np.float32)
                if w.sum() > 0:
                    manual_weights = w / (w.sum() + 1e-8)
            except Exception:
                manual_weights = None

        # validate
        if not processed_dir.exists():
            _emit(socketio, "ensemble_error", {"message": f"processed_dir not found: {processed_dir}"})
            return None
        if len(run_ids_in) != 3:
            _emit(socketio, "ensemble_error", {"message": "Please select EXACTLY 3 runs for ensemble."})
            return None

        run_ids = []
        for r in run_ids_in:
            sr = _safe_run_id(str(r))
            if not sr:
                _emit(socketio, "ensemble_error", {"message": f"Invalid run_id: {r}"})
                return None
            run_ids.append(sr)

        _emit(socketio, "ensemble_log", {"message": f"🧩 Ensemble start | dataset={dataset} | method={method}"})
        _emit(socketio, "ensemble_log", {"message": f"📁 processed_dir: {str(processed_dir)}"})
        _emit(socketio, "ensemble_log", {"message": f"🧠 runs: {run_ids}"})

        dataset_base = _infer_dataset_base_from_processed_dir(processed_dir)
        runs_root = dataset_base / "runs"
        if not runs_root.exists():
            _emit(socketio, "ensemble_error", {"message": f"runs folder not found: {runs_root}"})
            return None

        # load metrics + resolve weights
        metrics_list = []
        run_dirs = []
        weights_paths = []
        backbones = []

        for rid in run_ids:
            rd = runs_root / rid
            if not rd.exists():
                _emit(socketio, "ensemble_error", {"message": f"Run folder not found: {rd}"})
                return None

            m = _load_run_metrics(rd)
            metrics_list.append(m)
            run_dirs.append(rd)

            wp = _resolve_weights_path(rd, m)
            if wp is None or not wp.exists():
                # this is the exact issue you hit: run exists but weights file missing :contentReference[oaicite:6]{index=6}
                _emit(socketio, "ensemble_error", {
                    "message": f"No weights file found for run: {rid}. "
                               f"Expected best.weights.h5 in {rd}. "
                               f"Fix: patch trainer_three_cnn.py to always save fallback weights, then retrain this backbone."
                })
                return None
            weights_paths.append(wp)

            backbones.append(_extract_backbone(m, rid))

        # decide ensemble weights
        if manual_weights is not None:
            w = manual_weights
            _emit(socketio, "ensemble_log", {"message": f"⚖️ weights: {w.tolist()} (manual)"} )
        else:
            w = _auto_weights(weighting_mode, metrics_list)
            _emit(socketio, "ensemble_log", {"message": f"⚖️ weights: {w.tolist()} (auto={weighting_mode})"} )

        # decide image_size/num_classes from first run meta (must match across runs)
        meta0 = _extract_run_meta(metrics_list[0], payload)
        image_size = meta0["image_size"]
        num_classes = int(meta0["num_classes"] or 0)
        dropout = float(meta0["dropout"])
        seed = int(meta0["seed"])

        # sanity: enforce same image_size & num_classes
        for i in range(1, 3):
            mi = _extract_run_meta(metrics_list[i], payload)
            if tuple(mi["image_size"]) != tuple(image_size):
                _emit(socketio, "ensemble_error", {
                    "message": f"image_size mismatch across runs. "
                               f"Run1={image_size}, Run{i+1}={mi['image_size']}. "
                               f"Use runs trained with the same img_size."
                })
                return None
            if int(mi["num_classes"] or 0) != int(num_classes):
                _emit(socketio, "ensemble_error", {
                    "message": f"num_classes mismatch across runs. "
                               f"Run1={num_classes}, Run{i+1}={mi['num_classes']}. "
                               f"Use runs trained on the same dataset/classes."
                })
                return None

        _emit(socketio, "ensemble_log", {"message": f"🧱 Rebuilding models (image_size={image_size}, num_classes={num_classes}, dropout={dropout})"} )

        # build models and load weights
        models = []
        for bb, wp in zip(backbones, weights_paths):
            _emit(socketio, "ensemble_log", {"message": f"📦 Build+load: {bb} | weights={wp.name}"} )
            model = tcnn.build_model(
                backbone_name=str(bb),
                image_size=tuple(image_size),
                num_classes=int(num_classes),
                dropout=float(dropout),
                seed=int(seed),
            )
            model.load_weights(str(wp))
            models.append(model)

        # build dataset using the SAME split logic as training
        _emit(socketio, "ensemble_log", {"message": "📚 Building val dataset (same split as training)."} )
        _train_ds, val_ds, class_names, _steps = tcnn.make_datasets(
            processed_dir=str(processed_dir),
            image_size=tuple(image_size),          # IMPORTANT: use run metrics image_size, not hardcoded 224 :contentReference[oaicite:7]{index=7}
            batch_size=batch_size,
            cache=cache_dataset,
            seed=seed,
            socketio=socketio,
        )

        if len(class_names) <= 0:
            raise RuntimeError("No classes found in dataset folder.")

        # predict
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        _emit(socketio, "ensemble_log", {"message": "🔮 Predicting ensemble on val set."})
        for xb, yb in val_ds:
            y_np = yb.numpy() if hasattr(yb, "numpy") else np.array(yb)
            true_cls = np.argmax(y_np, axis=1).astype(int) if _is_one_hot(y_np) else y_np.astype(int)

            if method == "hardvote":
                preds = []
                for m in models:
                    probs = m.predict(xb, verbose=0)
                    preds.append(np.argmax(probs, axis=1).astype(int))
                # majority vote
                preds = np.stack(preds, axis=0)  # [3, B]
                # mode along axis 0
                pred_cls = np.apply_along_axis(lambda v: np.bincount(v, minlength=len(class_names)).argmax(), 0, preds)
            else:
                probs_sum = None
                for i, m in enumerate(models):
                    probs = m.predict(xb, verbose=0)
                    probs_sum = probs * w[i] if probs_sum is None else probs_sum + probs * w[i]
                pred_cls = np.argmax(probs_sum, axis=1).astype(int)

            y_true_all.extend(true_cls.tolist())
            y_pred_all.extend(pred_cls.tolist())

        # save artifacts
        ens_root = dataset_base / "ensemble_runs"
        ens_root.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{method}"
        out_dir = ens_root / run_id

        meta_out = {
            "dataset": dataset,
            "method": method,
            "weighting": ("manual" if manual_weights is not None else weighting_mode),
            "weights": [float(x) for x in w.tolist()],
            "run_ids": run_ids,
            "backbones": backbones,
            "processed_dir": str(processed_dir),
            "image_size": list(image_size),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "weights_files": [str(p) for p in weights_paths],
        }

        res = _save_ensemble_artifacts(out_dir, class_names, y_true_all, y_pred_all, meta_out)

        elapsed = time.time() - t0
        _emit(socketio, "ensemble_log", {
            "message": f"✅ Ensemble finished | acc={res['accuracy']:.4f} | QWK={res['kappa_qwk']:.4f} | time={elapsed:.1f}s"
        })

        summary = {
            "status": "ok",
            "run_dir": str(out_dir),
            "run_id": run_id,
            "dataset": dataset,
            "accuracy": res["accuracy"],
            "kappa_qwk": res["kappa_qwk"],
            "meta": meta_out,
        }

        _emit(socketio, "ensemble_done", {"summary": summary, "run_dir": str(out_dir)})
        return summary

    except Exception as e:
        _emit(socketio, "ensemble_error", {"message": str(e)})
        _emit(socketio, "ensemble_error", {"trace": traceback.format_exc()})
        return None
