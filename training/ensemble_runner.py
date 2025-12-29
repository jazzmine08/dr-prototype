# training/ensemble_runner.py
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score

import training.training_cnn_v2 as tcnn


CM_DPI = 170


def _emit(socketio, event: str, payload: dict):
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


def _safe_run_id(run_id: str) -> str:
    run_id = (run_id or "").strip()
    if not run_id:
        return ""
    bad = ("..", "/", "\\", ":", "\0")
    if any(x in run_id for x in bad):
        return ""
    return run_id


def _resolve_processed_final(processed_dir: Path) -> Path:
    p = Path(processed_dir).resolve()
    if p.name.lower() in ("train", "val", "test"):
        return p.parent
    if p.name.lower() == "processed_final":
        return p
    for parent in [p] + list(p.parents):
        if parent.name.lower() == "processed_final":
            return parent
    return p


def _infer_dataset_base_from_processed_final(processed_final: Path) -> Path:
    p = Path(processed_final).resolve()
    return p.parent if p.name.lower() == "processed_final" else p.parent


def _resolve_best_model_path(run_dir: Path) -> Optional[Path]:
    p = run_dir / "best_model.keras"
    return p if p.exists() else None


def _resolve_weights_path(run_dir: Path) -> Optional[Path]:
    p = run_dir / "best.weights.h5"
    if p.exists():
        return p
    # fallback: any weights
    for cand in run_dir.glob("*.weights.h5"):
        return cand
    return None


def _load_model_keras(best_model_path: Path):
    # Must provide custom_objects so .keras loads on another process
    custom_objects = {}
    if hasattr(tcnn, "BackbonePreprocess"):
        custom_objects["BackbonePreprocess"] = tcnn.BackbonePreprocess
    if hasattr(tcnn, "RandomBrightness"):
        custom_objects["RandomBrightness"] = tcnn.RandomBrightness

    return tcnn.keras.models.load_model(str(best_model_path), compile=False, custom_objects=custom_objects)


def _auto_weights(mode: str, metrics_list: List[dict]) -> np.ndarray:
    mode = (mode or "by_qwk").strip().lower()
    if mode not in ("equal", "by_qwk", "by_valacc"):
        mode = "by_qwk"

    if mode == "equal":
        w = np.ones(len(metrics_list), dtype=np.float32)
        return w / (w.sum() + 1e-8)

    key = "kappa_qwk" if mode == "by_qwk" else "best_val_accuracy"
    scores = []
    for m in metrics_list:
        v = m.get(key, None)
        if v is None:
            # try nested legacy style
            v = (m.get("summary") or {}).get(key, 0.0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        scores.append(max(0.0, v))

    arr = np.array(scores, dtype=np.float32)
    if arr.sum() <= 0:
        arr = np.ones(len(metrics_list), dtype=np.float32)
    return arr / (arr.sum() + 1e-8)


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: Path):
    cm = np.asarray(cm, dtype=np.int64)
    n = len(class_names)

    fig_w = max(6.6, 1.2 * n)
    fig_h = max(6.0, 1.0 * n)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    thresh = (cm.max() / 2.0) if cm.size else 0
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            ax.text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=CM_DPI)
    plt.close(fig)


def _build_eval_ds(eval_dir: Path, image_size: Tuple[int, int], batch_size: int, cache_dataset: bool, socketio=None):
    import tensorflow as tf

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval folder not found: {eval_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        str(eval_dir),
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    class_names = list(ds.class_names)

    def _cast(x, y):
        return tf.cast(x, tf.float32), y

    ds = ds.map(_cast, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_dataset:
        ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    _emit(socketio, "ensemble_log", {
        "message": f"✅ Eval dataset ready: dir={str(eval_dir)} | classes={class_names} | batches≈{len(ds)}"
    })
    return ds, class_names


def run_ensemble(socketio, payload: Dict[str, Any]):
    """
    Writes:
      - ensemble_runs/<timestamp>_.../metrics.json   (WITH kappa_qwk, best_val_accuracy, best_val_loss)
      - ensemble_runs/<timestamp>_.../evaluation.json (WITH accuracy)
      - report.txt, confusion_matrix.png, predictions.json
    So results.html will show:
      QWK, Best Val Acc, Best Val Loss, Eval Accuracy
    """
    t0 = time.time()
    try:
        dataset = (payload.get("dataset") or "unknown").strip().lower()

        processed_dir_raw = payload.get("processed_dir") or payload.get("processedDir") or ""
        processed_final = _resolve_processed_final(Path(str(processed_dir_raw)))

        eval_split = (payload.get("eval_split") or "val").strip().lower()
        if eval_split in ("validation", "valid"):
            eval_split = "val"
        if eval_split not in ("val", "test"):
            eval_split = "val"

        method = (payload.get("method") or "softvote").strip().lower()
        weighting_mode = (payload.get("weighting") or "by_qwk").strip().lower()
        if weighting_mode not in ("equal", "by_qwk", "by_valacc"):
            weighting_mode = "by_qwk"

        batch_size = int(payload.get("batch_size", payload.get("batchSize", 16)) or 16)
        cache_dataset = bool(payload.get("cache_dataset", payload.get("cacheDataset", False)))

        run_ids_in = payload.get("run_ids") or payload.get("runIds") or []
        if not (isinstance(run_ids_in, list) and len(run_ids_in) == 3):
            r1 = (payload.get("run1") or "").strip()
            r2 = (payload.get("run2") or "").strip()
            r3 = (payload.get("run3") or "").strip()
            if r1 and r2 and r3:
                run_ids_in = [r1, r2, r3]

        if not (isinstance(run_ids_in, list) and len(run_ids_in) == 3):
            _emit(socketio, "ensemble_error", {"message": "Please select EXACTLY 3 runs for ensemble."})
            return None

        run_ids = []
        for r in run_ids_in:
            sr = _safe_run_id(str(r))
            if not sr:
                _emit(socketio, "ensemble_error", {"message": f"Invalid run_id: {r}"})
                return None
            run_ids.append(sr)

        if not processed_final.exists():
            _emit(socketio, "ensemble_error", {"message": f"processed_final not found: {processed_final}"})
            return None

        dataset_base = _infer_dataset_base_from_processed_final(processed_final)
        runs_root = dataset_base / "runs"
        if not runs_root.exists():
            _emit(socketio, "ensemble_error", {"message": f"runs folder not found: {runs_root}"})
            return None

        _emit(socketio, "ensemble_log", {"message": f"🧩 Ensemble start | dataset={dataset} | eval_split={eval_split} | method={method} | weighting={weighting_mode}"})
        _emit(socketio, "ensemble_log", {"message": f"📁 processed_final: {processed_final}"})
        _emit(socketio, "ensemble_log", {"message": f"🧠 runs: {run_ids}"})

        # Load 3 models (prefer best_model.keras)
        models = []
        metrics_list = []
        sources = []

        for rid in run_ids:
            rd = runs_root / rid
            if not rd.exists():
                _emit(socketio, "ensemble_error", {"message": f"Run folder not found: {rd}"})
                return None

            mp = _resolve_best_model_path(rd)
            if mp is not None:
                _emit(socketio, "ensemble_log", {"message": f"📦 Loading model: {rd.name} | {mp.name}"})
                model = _load_model_keras(mp)
                models.append(model)
                sources.append(str(mp))
            else:
                # fallback (still can fail if architecture differs)
                wp = _resolve_weights_path(rd)
                if wp is None:
                    _emit(socketio, "ensemble_error", {"message": f"No best_model.keras and no weights found for run: {rd.name}"})
                    return None
                _emit(socketio, "ensemble_log", {"message": f"⚠️ best_model.keras missing; fallback to weights: {rd.name} | {wp.name}"})

                # build a model using the same backbone name stored in metrics.json (or infer from folder)
                mjson = _read_json(rd / "metrics.json")
                bb = (mjson.get("backbone") or mjson.get("model") or "").strip() or rd.name.split("_", 1)[-1]
                img_size = int((mjson.get("extra") or {}).get("image_size", [224, 224])[0] if isinstance((mjson.get("extra") or {}).get("image_size"), list) else mjson.get("img_size", 224))
                num_classes = int(mjson.get("num_classes") or 5)

                cfg = tcnn.TrainConfig(
                    dataset=dataset,
                    processed_dir=processed_final,
                    batch_size=batch_size,
                    img_size=img_size,
                    dropout=float((mjson.get("extra") or {}).get("dropout", 0.3)),
                    seed=int((mjson.get("extra") or {}).get("seed", 123)),
                    cache_dataset=cache_dataset,
                    use_augmentation=False,
                )
                model, _ = tcnn.build_model(bb, (img_size, img_size), num_classes, cfg)
                model.load_weights(str(wp))
                models.append(model)
                sources.append(str(wp))

            # keep metrics for weighting
            mjson = _read_json((runs_root / rid) / "metrics.json")
            metrics_list.append(mjson if mjson else {})

        # Sanity check: input/output consistency
        in_sizes = []
        out_dims = []
        for m in models:
            in_sizes.append(tuple(m.input_shape[1:3]))
            out_dims.append(int(m.output_shape[-1]))

        if any(s != in_sizes[0] for s in in_sizes):
            _emit(socketio, "ensemble_error", {"message": f"Model input sizes mismatch across runs: {in_sizes}. Train all with same img_size."})
            return None
        if any(d != out_dims[0] for d in out_dims):
            _emit(socketio, "ensemble_error", {"message": f"Model output dims mismatch across runs: {out_dims}. Train all with same num_classes."})
            return None

        image_size = (int(in_sizes[0][0]), int(in_sizes[0][1]))
        num_classes = int(out_dims[0])

        # weights for soft-vote
        w = _auto_weights(weighting_mode, metrics_list)

        _emit(socketio, "ensemble_log", {"message": f"✅ Models ready | image_size={image_size} | num_classes={num_classes} | weights={w.tolist()}"})


        # Eval dataset
        eval_dir = processed_final / eval_split
        eval_ds, class_names = _build_eval_ds(eval_dir, image_size=image_size, batch_size=batch_size, cache_dataset=cache_dataset, socketio=socketio)

        # Predict
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        _emit(socketio, "ensemble_log", {"message": f"🔮 Predicting ensemble on {eval_split} set..."})
        for xb, yb in eval_ds:
            y_np = yb.numpy() if hasattr(yb, "numpy") else np.array(yb)
            true_cls = np.argmax(y_np, axis=1).astype(int)
            y_true_all.extend(true_cls.tolist())

            if method in ("hardvote", "hardvote_majority", "majority"):
                preds = []
                for m in models:
                    probs = m.predict(xb, verbose=0)
                    preds.append(np.argmax(probs, axis=1).astype(int))
                preds = np.stack(preds, axis=0)
                pred_cls = np.apply_along_axis(lambda v: np.bincount(v, minlength=len(class_names)).argmax(), 0, preds)
            else:
                probs_sum = None
                for i, m in enumerate(models):
                    probs = m.predict(xb, verbose=0)
                    probs_sum = probs * w[i] if probs_sum is None else probs_sum + probs * w[i]
                pred_cls = np.argmax(probs_sum, axis=1).astype(int)

            y_pred_all.extend(pred_cls.tolist())

        # Metrics (these are your ensemble “Eval Accuracy” + “QWK”)
        acc = float(accuracy_score(y_true_all, y_pred_all))
        qwk = float(cohen_kappa_score(y_true_all, y_pred_all, weights="quadratic"))

        # Save outputs
        ens_root = dataset_base / "ensemble_runs"
        ens_root.mkdir(parents=True, exist_ok=True)

        run_id = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{method}_{weighting_mode}_{eval_split}"
        out_dir = ens_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(class_names))))
        report = classification_report(
            y_true_all, y_pred_all,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
        (out_dir / "report.txt").write_text(report, encoding="utf-8")
        _plot_confusion_matrix(cm, class_names, "Ensemble Confusion Matrix", out_dir / "confusion_matrix.png")

        # ✅ evaluation.json (Results page uses evaluation.accuracy)
        evaluation = {
            "accuracy": acc,
            "kappa_qwk": qwk,
            "num_samples": int(len(y_true_all)),
        }
        (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

        # ✅ metrics.json (Results page uses metrics.kappa_qwk, metrics.best_val_accuracy, metrics.best_val_loss)
        # Ensemble has no training loss, so best_val_loss is N/A -> None
        metrics = {
            "dataset": dataset,
            "model": "Ensemble",
            "backbone": "Ensemble",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_id": run_id,
            "run_dir": str(out_dir),

            "kappa_qwk": qwk,
            "best_val_accuracy": acc,      # show ensemble accuracy here (UI expects it)
            "best_val_loss": None,         # ensemble has no val_loss
            "num_val_samples": int(len(y_true_all)),

            "artifacts": {
                "confusion_matrix.png": str(out_dir / "confusion_matrix.png"),
                "report.txt": str(out_dir / "report.txt"),
                "evaluation.json": str(out_dir / "evaluation.json"),
                "predictions.json": str(out_dir / "predictions.json"),
            },
            "extra": {
                "eval_split": eval_split,
                "eval_dir": str(eval_dir),
                "method": method,
                "weighting": weighting_mode,
                "weights": [float(x) for x in w.tolist()],
                "run_ids": run_ids,
                "model_sources": sources,
                "image_size": list(image_size),
                "num_classes": int(num_classes),
                "batch_size": int(batch_size),
            },
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        (out_dir / "predictions.json").write_text(
            json.dumps({"y_true": y_true_all, "y_pred": y_pred_all}, indent=2),
            encoding="utf-8"
        )

        elapsed = time.time() - t0
        _emit(socketio, "ensemble_log", {"message": f"✅ Ensemble finished | acc={acc:.4f} | QWK={qwk:.4f} | time={elapsed:.1f}s"})
        _emit(socketio, "ensemble_done", {"summary": {"status": "ok", "run_dir": str(out_dir), "accuracy": acc, "kappa_qwk": qwk}, "run_dir": str(out_dir)})

        return {"status": "ok", "run_dir": str(out_dir), "accuracy": acc, "kappa_qwk": qwk}

    except Exception as e:
        _emit(socketio, "ensemble_error", {"message": str(e)})
        _emit(socketio, "ensemble_error", {"trace": traceback.format_exc()})
        return None
