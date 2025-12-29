# training/trainer_three_cnn.py
# Orchestrates training of 3 backbones and guarantees:
#   - consistent "balanced_sampling" flag in both config + record
#   - outputs always written inside the SAME local dataset folder:
#       X:\dr_prototype\processed\<dataset>\runs\<run_id>\...
#
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import training.training_cnn_v2 as tcnn

BACKBONES = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]


def _emit(socketio, event: str, payload: dict):
    """Safe emit (won't crash if socketio is None)."""
    try:
        if socketio is not None:
            socketio.emit(event, payload)
        else:
            print(f"[{event}] {payload}")
    except Exception:
        pass


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert objects into JSON-serializable primitives.
    - Path -> str
    - numpy scalars -> python scalars (if numpy is present)
    - dict/list/tuple -> recurse
    - everything else -> as-is (or str fallback for unknown objects)
    """
    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]):
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _resolve_dataset_base(processed_dir: Path) -> Path:
    """
    processed_dir is usually:
      ...\processed\<dataset>\processed_final\train
    We return:
      ...\processed\<dataset>
    """
    p = processed_dir.resolve()

    if p.name.lower() == "train":
        # processed_final/train -> processed/<dataset>
        return p.parent.parent

    if p.name.lower() == "processed_final":
        # processed_final -> processed/<dataset>
        return p.parent

    # fallback: climb until we find processed_final
    for parent in [p] + list(p.parents):
        if parent.name.lower() == "processed_final":
            return parent.parent

    return p.parent


def run_cnn_training(socketio, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry used by Flask/SocketIO. Returns a JSON-serializable dict."""
    t0 = time.time()

    dataset = str(payload.get("dataset", "")).strip().lower()
    processed_dir = Path(payload.get("processed_dir", "")).expanduser()

    if not dataset:
        raise ValueError("payload.dataset is required")
    if not str(processed_dir):
        raise ValueError("payload.processed_dir is required (point to .../processed_final/train or .../processed_final)")

    cfg = tcnn.TrainConfig(
        dataset=dataset,
        processed_dir=processed_dir,
        epochs=int(payload.get("epochs", 6)),
        warmup_epochs=int(payload.get("warmup_epochs", 1)),
        batch_size=int(payload.get("batch_size", 16)),
        img_size=int(payload.get("img_size", 224)),
        lr=float(payload.get("lr", 1e-4)),
        fine_tune_lr=float(payload.get("fine_tune_lr", 1e-5)),
        dropout=float(payload.get("dropout", 0.3)),
        seed=int(payload.get("seed", 123)),
        cache_dataset=bool(payload.get("cache_dataset", False)),

        balanced_sampling=bool(payload.get("balanced_sampling", False)),

        use_augmentation=bool(payload.get("use_augmentation", True)),
        aug_flip=str(payload.get("aug_flip", "horizontal")),
        aug_rot_deg=float(payload.get("aug_rot_deg", 30.0)),
        aug_zoom=float(payload.get("aug_zoom", 0.20)),
        aug_shift=float(payload.get("aug_shift", 0.05)),
        aug_contrast=float(payload.get("aug_contrast", 0.20)),
        aug_brightness=float(payload.get("aug_brightness", 0.15)),
        aug_noise_std=float(payload.get("aug_noise_std", 0.00)),

        force_stratified_split=bool(payload.get("force_stratified_split", False)),
        val_ratio=float(payload.get("val_ratio", 0.20)),

        save_history=bool(payload.get("save_history", True)),
    )

    dataset_base = _resolve_dataset_base(cfg.processed_dir)
    runs_root = dataset_base / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    _emit(socketio, "train_log", {"message": f"✅ Dataset: {dataset} | img_size={cfg.img_size} | batch={cfg.batch_size}"})
    _emit(socketio, "train_log", {"message": f"📌 Outputs -> {runs_root} (local)"})

    results: List[Dict[str, Any]] = []
    overall_ok = True

    # Pre-convert config once (Path-safe)
    cfg_dict = _json_safe(asdict(cfg))

    for backbone in BACKBONES:
        run_id = f"{_timestamp()}_{backbone}"
        run_dir = runs_root / run_id

        _emit(socketio, "train_log", {"message": f"\n▶️ Training {backbone} | run_id={run_id}"})

        try:
            out = tcnn.train_one_backbone(cfg=cfg, backbone_name=backbone, run_dir=run_dir, socketio=socketio)

            record: Dict[str, Any] = {
                "status": "ok",
                "dataset": dataset,
                "model": backbone,
                "backbone": backbone,
                "created_at": out.get("created_at"),
                "run_id": run_id,
                "run_dir": str(run_dir),

                "best_val_accuracy": out.get("best_val_accuracy"),
                "kappa_qwk": out.get("kappa_qwk"),
                "num_val_samples": out.get("num_val_samples"),

                "best_weights_path": out.get("best_weights_path"),
                "best_model_path": out.get("best_model_path"),
                "evaluation_path": out.get("evaluation_path"),

                "balanced_sampling": bool(cfg.balanced_sampling),
                "balanced_info": out.get("balanced_info"),
                "use_augmentation": bool(cfg.use_augmentation),
                "force_stratified_split": bool(cfg.force_stratified_split),
                "val_ratio": float(cfg.val_ratio),

                "config": cfg_dict,
            }

            _write_json(run_dir / "run_record.json", record)
            results.append(_json_safe(record))

            acc_val = float(record["best_val_accuracy"] or 0.0)
            qwk_val = float(record["kappa_qwk"] or 0.0)
            _emit(socketio, "train_log", {"message": f"✅ Done {backbone} | acc={acc_val:.4f} | qwk={qwk_val:.4f}"})

        except Exception as e:
            overall_ok = False
            tb = traceback.format_exc()
            _emit(socketio, "train_log", {"message": f"❌ ERROR training {backbone}: {e}\n{tb}"})

            err_record: Dict[str, Any] = {
                "status": "error",
                "dataset": dataset,
                "model": backbone,
                "backbone": backbone,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "error": str(e),
                "traceback": tb,
                "config": cfg_dict,
            }

            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                _write_json(run_dir / "run_record.json", err_record)
            except Exception:
                pass

            results.append(_json_safe(err_record))

    elapsed = time.time() - t0

    final_payload = _json_safe({
        "status": "ok" if overall_ok else "partial",
        "dataset": dataset,
        "elapsed_sec": float(elapsed),
        "results": results,
    })

    # ✅ What your UI likely listens for
    _emit(socketio, "train_done", {"summary": final_payload})
    _emit(socketio, "train_log", {"message": f"🏁 Training finished | status={final_payload['status']} | time={elapsed:.1f}s"})

    return final_payload
