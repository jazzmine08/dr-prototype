# training/trainer_three_cnn.py
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

import training.training_cnn_v2 as tcnn
from training.results_utils import make_run_dir, save_training_artifacts


def _emit(socketio, event: str, payload: dict):
    """Safe emit (won't crash if socketio is None)."""
    try:
        if socketio is not None:
            socketio.emit(event, payload)
        else:
            # fallback: print so you still get logs in console
            print(f"[{event}] {payload}")
    except Exception:
        # don't kill training due to logging
        pass


# ---------------------------
# store a global results file (optional)
# ---------------------------
def _append_to_full_results(results_dir: str, record: dict):
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, "cnn", "training_results_full.json")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    data = []
    if os.path.exists(full_path):
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []

    data.append(record)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _infer_dataset_name_from_processed_dir(processed_dir: str) -> str:
    """
    processed_dir example:
      G:\\My Drive\\dr_prototype\\processed\\aptos2019\\processed_final\\train
    dataset_name = 'aptos2019' (parent of processed_final)
    """
    try:
        p = Path(processed_dir).resolve()
        return p.parents[1].name
    except Exception:
        return "unknown"


def _train_one_backbone(socketio, cfg: tcnn.TrainConfig, backbone_name: str, dataset_name: str) -> dict:
    # 1) Build datasets
    train_ds, val_ds, class_names, steps_per_epoch = tcnn.make_datasets(
        processed_dir=cfg.processed_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        cache=cfg.cache_dataset,
        seed=cfg.seed,
        socketio=socketio,
    )

    num_classes = len(class_names)
    if num_classes <= 0:
        raise RuntimeError("No classes found in dataset folder. Check processed_dir structure.")

    # 2) Create run folder (SINGLE SOURCE OF TRUTH: make_run_dir in results_utils.py)
    run_dir = make_run_dir(cfg.processed_dir, dataset_name, backbone_name)

    _emit(socketio, "training_log", {"message": f"🔧 Training ONE model: {backbone_name} | dataset={dataset_name}"})
    _emit(socketio, "training_log", {"message": f"📁 Run folder: {run_dir}"})

    # 3) Build model
    model = tcnn.build_model(
        backbone_name=backbone_name,
        image_size=cfg.image_size,
        num_classes=num_classes,
        dropout=cfg.dropout,
        seed=cfg.seed,
    )

    # 4) Callbacks
    # IMPORTANT: weights-only checkpoint (avoids Lambda deserialization issues)
    best_weights_path = run_dir / "best.weights.h5"
    log_csv_path = run_dir / "train_log.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=False),
        CSVLogger(str(log_csv_path)),
        tcnn.SocketProgressCallback(socketio, backbone_name, cfg.epochs, steps_per_epoch),
    ]

    t0 = time.time()
    hist_all: Dict[str, List[float]] = {}

    def _merge(h):
        for k, v in h.history.items():
            hist_all.setdefault(k, [])
            hist_all[k].extend([float(x) for x in v])

    # 5) Stage 1 (warmup)
    warmup = min(cfg.warmup_epochs, cfg.epochs)
    if hasattr(model, "backbone"):
        model.backbone.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(cfg.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if warmup > 0:
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warmup,
            callbacks=callbacks,
            verbose=0,
        )
        _merge(h1)

    # 6) Stage 2 (fine-tune)
    remaining = cfg.epochs - warmup
    if remaining > 0:
        if hasattr(model, "backbone"):
            model.backbone.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(cfg.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        h2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
        )
        _merge(h2)

    elapsed = time.time() - t0

    # 7) Rebuild + load best weights (NO load_model -> avoids Lambda/serialization issues)
    best_model = tcnn.build_model(
        backbone_name=backbone_name,
        image_size=cfg.image_size,
        num_classes=num_classes,
        dropout=cfg.dropout,
        seed=cfg.seed,
    )

    if best_weights_path.exists():
        best_model.load_weights(str(best_weights_path))
    else:
        # fallback: if checkpoint wasn't written for any reason, use current model weights
        best_model.set_weights(model.get_weights())

    # compile is optional for predict-based eval, but safe to keep
    best_model.compile(
        optimizer=keras.optimizers.Adam(cfg.fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 8) Save artifacts + metrics.json inside run_dir
    artifacts_summary = save_training_artifacts(
        run_dir=run_dir,
        history=hist_all,
        model=best_model,
        val_ds=val_ds,
        class_names=class_names,
        meta={
            "dataset": dataset_name,
            "backbone": backbone_name,
            "train_time_s": round(elapsed, 2),
            "best_weights_path": str(best_weights_path),
            "processed_dir": cfg.processed_dir,
            "num_classes": num_classes,
            "class_names": class_names,
            "epochs": cfg.epochs,
            "warmup_epochs": cfg.warmup_epochs,
            "batch_size": cfg.batch_size,
            "image_size": list(cfg.image_size) if isinstance(cfg.image_size, tuple) else cfg.image_size,
            "lr": cfg.lr,
            "fine_tune_lr": cfg.fine_tune_lr,
            "dropout": cfg.dropout,
            "seed": cfg.seed,
        },
    )

    _emit(socketio, "training_log", {"message": f"📊 Saved artifacts to: {run_dir}"})

    # 9) Optional evaluation.json
    eval_payload = tcnn.evaluate_model(best_model, val_ds)
    eval_payload.update({
        "model": backbone_name,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "best_weights_path": str(best_weights_path),
    })

    eval_path = run_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    best_val_acc = float(max(hist_all.get("val_accuracy", [0.0])))

    record = {
        "dataset": dataset_name,
        "model": backbone_name,
        "accuracy": float(eval_payload.get("accuracy", 0.0)),
        "f1_macro": float(eval_payload.get("f1_macro", 0.0)),
        "precision_macro": float(eval_payload.get("precision_macro", 0.0)),
        "recall_macro": float(eval_payload.get("recall_macro", 0.0)),
        "kappa_qwk": artifacts_summary.get("kappa_qwk"),
        "best_val_accuracy": best_val_acc,
        "train_time_s": round(elapsed, 2),
        "best_weights_path": str(best_weights_path),
        "evaluation_path": str(eval_path),
        "run_dir": str(run_dir),
        "created_at": eval_payload["created_at"],
        "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else cfg.__dict__,
    }

    _append_to_full_results(cfg.results_dir, record)

    _emit(socketio, "training_log", {
        "message": f"✅ Done {backbone_name} — best_val={best_val_acc:.4f} — "
                   f"eval_acc={record['accuracy']:.4f} — QWK={record.get('kappa_qwk')}"
    })

    _emit(socketio, "training_done", {
        "status": "ok",
        "results": [record],
        "run_dir": str(run_dir),
        "run_id": Path(run_dir).name,
        "dataset": dataset_name,
        "model": backbone_name
    })
    return record

def _normalize_backbone_name(name: str) -> str:
    if not name:
        return name
    n = str(name).strip().lower().replace("-", "").replace("_", "")
    alias = {
        "efficientnet": "EfficientNetV2B0",
        "efficientnetv2b0": "EfficientNetV2B0",
        "densenet": "DenseNet121",
        "densenet121": "DenseNet121",
        "inceptionresnet": "InceptionResNetV2",
        "inceptionresnetv2": "InceptionResNetV2",
    }
    return alias.get(n, str(name).strip())


def run_cnn_training(socketio, payload: dict):
    """
    Run training:
    - If payload["model"] is provided -> train only that model
    - Else -> train the 3 required backbones sequentially
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        results_dir = os.path.join(models_dir, "results")

        processed_dir = payload.get("processed_dir")
        if not processed_dir or not os.path.isdir(processed_dir):
            _emit(socketio, "training_error", {"message": f"processed_dir not found: {processed_dir}"})
            return None

        dataset_name = (payload.get("dataset") or _infer_dataset_name_from_processed_dir(processed_dir)).strip().lower()

        cfg = tcnn.TrainConfig(
            processed_dir=processed_dir,
            models_dir=models_dir,
            results_dir=results_dir,
            epochs=int(payload.get("epochs", 20)),
            warmup_epochs=int(payload.get("warmup_epochs", 2)),
            batch_size=int(payload.get("batch_size", 16)),
            image_size=(int(payload.get("img_size", 224)), int(payload.get("img_size", 224))),
            lr=float(payload.get("lr", 1e-4)),
            fine_tune_lr=float(payload.get("fine_tune_lr", 1e-5)),
            dropout=float(payload.get("dropout", 0.3)),
            cache_dataset=bool(payload.get("cache_dataset", False)),
            seed=int(payload.get("seed", 123)),
        )

        _emit(socketio, "training_log", {"message": f"✅ Dataset: {dataset_name}"})

        # Train single model if provided
        model_name = payload.get("model")
        if model_name:
            canon = _normalize_backbone_name(model_name)
            return _train_one_backbone(socketio, cfg, canon, dataset_name)



        # Train the three models in your objective (fallback)
        backbones = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]

        results = []
        for b in backbones:
            try:
                results.append(_train_one_backbone(socketio, cfg, str(b), dataset_name))
            except Exception as e:
                _emit(socketio, "training_error", {"message": f"❌ Failed backbone {b}: {e}"})
                _emit(socketio, "training_error", {"trace": traceback.format_exc()})

        _emit(socketio, "training_done", {"status": "ok", "results": results})
        return results

    except Exception as e:
        _emit(socketio, "training_error", {"message": str(e)})
        _emit(socketio, "training_error", {"trace": traceback.format_exc()})
        return None
