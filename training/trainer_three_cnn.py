# training/trainer_three_cnn.py
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

import training.training_cnn_v2 as tcnn
from training.results_utils import make_run_dir, save_training_artifacts


# =========================================================
# Socket helpers
# =========================================================
def _emit(socketio, event: str, payload: dict):
    """Safe emit (won't crash if socketio is None)."""
    try:
        if socketio is not None:
            socketio.emit(event, payload)
        else:
            print(f"[{event}] {payload}")
    except Exception:
        pass


# =========================================================
# Full results append (safe + creates folders)
# =========================================================
def _append_to_full_results(results_dir: str, record: dict):
    """
    Appends one record into:
      <results_dir>/cnn/training_results_full.json

    NOTE:
    - results_dir passed in here is already something like: <project_root>/models/results
    """
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


# =========================================================
# Infer dataset name
# =========================================================
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


# =========================================================
# Class weights
# =========================================================
def _compute_class_weights_from_train_ds(train_ds, num_classes: int) -> Dict[int, float]:
    """
    train_ds labels are one-hot because make_datasets uses label_mode="categorical".
    """
    y_all: List[int] = []
    for _, y in train_ds:
        y_np = y.numpy()
        y_all.extend(np.argmax(y_np, axis=1).tolist())

    y_all = np.array(y_all, dtype=int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_all
    )
    return {int(i): float(w) for i, w in enumerate(weights)}


# =========================================================
# Backbone name normalize
# =========================================================
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


# =========================================================
# Train ONE backbone
# =========================================================
def _train_one_backbone(socketio, cfg: tcnn.TrainConfig, backbone_name: str, dataset_name: str) -> dict:
    backbone_name = _normalize_backbone_name(backbone_name)

    # 1) Build datasets (uses sibling val folder if processed_dir points to .../train)
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

    # 2) Class weights (imbalance handling)
    class_weight_dict = _compute_class_weights_from_train_ds(train_ds, num_classes)
    _emit(socketio, "training_log", {"message": f"⚖️ Using class_weight: {class_weight_dict}"})

    # 3) Create run folder (this function should place it under STORAGE_ROOT/<dataset>/runs)
    run_dir = make_run_dir(cfg.processed_dir, dataset_name, backbone_name)
    run_dir = Path(run_dir)

    _emit(socketio, "training_log", {"message": f"🔧 Training: {backbone_name} | dataset={dataset_name}"})
    _emit(socketio, "training_log", {"message": f"📁 Run folder: {run_dir}"})

    # 4) Build model
    model = tcnn.build_model(
        backbone_name=backbone_name,
        image_size=cfg.image_size,
        num_classes=num_classes,
        dropout=cfg.dropout,
        seed=cfg.seed,
    )

    # 5) Paths
    best_weights_path = run_dir / "best.weights.h5"
    last_weights_path = run_dir / "last.weights.h5"
    best_model_path = run_dir / "best_model.keras"
    signature_path = run_dir / "model_signature.json"
    log_csv_path = run_dir / "train_log.csv"

    # 6) Callbacks
    callbacks = [
        # Best weights
        ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=1,
        ),
        # Last weights (always overwritten)
        ModelCheckpoint(
            filepath=str(last_weights_path),
            monitor="val_accuracy",
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=False),
        CSVLogger(str(log_csv_path)),
        tcnn.SocketProgressCallback(socketio, backbone_name, cfg.epochs, steps_per_epoch),
    ]

    # 7) Train (warmup + finetune)
    t0 = time.time()
    hist_all: Dict[str, List[float]] = {}

    def _merge(h):
        for k, v in h.history.items():
            hist_all.setdefault(k, [])
            hist_all[k].extend([float(x) for x in v])

    warmup = min(int(cfg.warmup_epochs), int(cfg.epochs))

    # Stage 1
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
            class_weight=class_weight_dict,
        )
        _merge(h1)

    # Stage 2
    remaining = int(cfg.epochs) - warmup
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
            epochs=int(cfg.epochs),
            callbacks=callbacks,
            verbose=0,
            class_weight=class_weight_dict,
        )
        _merge(h2)

    elapsed = time.time() - t0

    # 8) Make sure best.weights.h5 exists (fallback to last)
    if not best_weights_path.exists():
        if last_weights_path.exists():
            try:
                import shutil
                shutil.copy2(str(last_weights_path), str(best_weights_path))
                _emit(socketio, "training_log", {"message": "⚠️ best.weights.h5 missing → copied from last.weights.h5"})
            except Exception as e:
                _emit(socketio, "training_log", {"message": f"⚠️ Could not copy last→best weights: {e}"})

    if not best_weights_path.exists():
        # last fallback: save current weights
        model.save_weights(str(best_weights_path))
        _emit(socketio, "training_log", {"message": "⚠️ best.weights.h5 missing → saved current weights as best.weights.h5"})

    # 9) Rebuild a clean model + load best weights (avoids mismatch issues)
    best_model = tcnn.build_model(
        backbone_name=backbone_name,
        image_size=cfg.image_size,
        num_classes=num_classes,
        dropout=cfg.dropout,
        seed=cfg.seed,
    )
    best_model.load_weights(str(best_weights_path))
    best_model.compile(
        optimizer=keras.optimizers.Adam(cfg.fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 10) Save model signature
    signature = {
        "dataset": dataset_name,
        "backbone": backbone_name,
        "image_size": list(cfg.image_size) if isinstance(cfg.image_size, tuple) else cfg.image_size,
        "dropout": float(cfg.dropout),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "best_weights_path": str(best_weights_path),
        "last_weights_path": str(last_weights_path),
        "best_model_path": str(best_model_path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(signature_path, "w", encoding="utf-8") as f:
            json.dump(signature, f, indent=2)
    except Exception as e:
        _emit(socketio, "training_log", {"message": f"⚠️ Could not save model_signature.json: {e}"})

    # 11) Save full model too (should be safe now: NO Lambda in training_cnn_v2.py)
    try:
        best_model.save(str(best_model_path), include_optimizer=False)
        _emit(socketio, "training_log", {"message": f"💾 Saved full model: {best_model_path.name}"})
    except Exception as e:
        _emit(socketio, "training_log", {"message": f"⚠️ Could not save full model (.keras): {e}"})

    # 12) Save artifacts (curves, confusion_matrix.png, metrics.json, report.txt, evaluation.json, etc.)
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
            "last_weights_path": str(last_weights_path),
            "best_model_path": str(best_model_path),
            "processed_dir": cfg.processed_dir,
            "num_classes": num_classes,
            "class_names": class_names,
            "epochs": int(cfg.epochs),
            "warmup_epochs": int(cfg.warmup_epochs),
            "batch_size": int(cfg.batch_size),
            "image_size": list(cfg.image_size) if isinstance(cfg.image_size, tuple) else cfg.image_size,
            "lr": float(cfg.lr),
            "fine_tune_lr": float(cfg.fine_tune_lr),
            "dropout": float(cfg.dropout),
            "seed": int(cfg.seed),
            "class_weight": class_weight_dict,
        },
    )
    _emit(socketio, "training_log", {"message": f"📊 Saved artifacts to: {run_dir}"})


    # 13) evaluation.json (explicit, in case you read it in other pages)
    eval_payload = tcnn.evaluate_model(best_model, val_ds)
    eval_payload.update({
        "model": backbone_name,
        "dataset": dataset_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": list(cfg.image_size) if isinstance(cfg.image_size, tuple) else cfg.image_size,
        "dropout": float(cfg.dropout),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "best_weights_path": str(best_weights_path),
        "last_weights_path": str(last_weights_path),
        "best_model_path": str(best_model_path),
    })

    eval_path = run_dir / "evaluation.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    # best val acc
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
        "last_weights_path": str(last_weights_path),
        "best_model_path": str(best_model_path),
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
        "run_id": run_dir.name,
        "dataset": dataset_name,
        "model": backbone_name
    })
    return record


# =========================================================
# Main entry called by app.py
# =========================================================
def run_cnn_training(socketio, payload: dict):
    """
    payload expected from app.py:
      dataset: "aptos2019" | "drtid"
      processed_dir: ".../processed_final/train"
      epochs, batch_size, img_size, warmup_epochs, lr, fine_tune_lr, dropout, cache_dataset, seed
      model: optional backbone name to train only one backbone

    Runs:
      - one backbone (if payload["model"] provided)
      - otherwise trains the 3 backbones sequentially
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
        img_size = int(payload.get("img_size", 224))

        cfg = tcnn.TrainConfig(
            processed_dir=str(processed_dir),
            models_dir=str(models_dir),
            results_dir=str(results_dir),
            epochs=int(payload.get("epochs", 20)),
            warmup_epochs=int(payload.get("warmup_epochs", 2)),
            batch_size=int(payload.get("batch_size", 16)),
            image_size=(img_size, img_size),
            lr=float(payload.get("lr", 1e-4)),
            fine_tune_lr=float(payload.get("fine_tune_lr", 1e-5)),
            dropout=float(payload.get("dropout", 0.3)),
            cache_dataset=bool(payload.get("cache_dataset", False)),
            seed=int(payload.get("seed", 123)),
        )

        _emit(socketio, "training_log", {"message": f"✅ Dataset: {dataset_name} | img_size={img_size} | batch={cfg.batch_size}"})

        model_name = payload.get("model")
        if model_name:
            canon = _normalize_backbone_name(model_name)
            return _train_one_backbone(socketio, cfg, canon, dataset_name)

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
