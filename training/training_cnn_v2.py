# training_cnn_v2.py
"""
Thesis training script (3 CNN backbones): DenseNet121, InceptionResNetV2, EfficientNetV2B0

Key fixes vs your current training_cnn.py:
- Works with or WITHOUT socketio (socketio is optional)
- Correct preprocessing per backbone (critical for ImageNet pretrained models)
- Optional 2-stage training: freeze backbone warmup -> fine-tune
- Evaluates using the best checkpoint (loads saved model)
- Writes ONE full results file + per-model evaluation files (no accidental overwrite)

Dataset layout supported:
A) processed_dir/train/<class>/*.jpg + processed_dir/val/<class>/*.jpg
B) processed_dir/<class>/*.jpg (auto 80/20 split)

Outputs:
- models_dir/cnn/<ModelName>.keras
- results_dir/cnn/<ModelName>_evaluation.json
- results_dir/cnn/training_results_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import traceback
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

AUTOTUNE = tf.data.AUTOTUNE


# -------------------------
# Socket helper (optional)
# -------------------------
def _emit(socketio, event: str, payload: Dict[str, Any]) -> None:
    if socketio is None:
        return
    try:
        socketio.emit(event, payload)
    except Exception:
        # never crash training because of UI
        pass


# -------------------------
# Filename normalization (optional but useful)
# -------------------------
def _safe_name(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._-]", "", name)
    return name or "file"


def normalize_and_clean_dir(root_dir: str, socketio=None) -> Dict[str, Any]:
    renamed, removed, total_checked = [], [], 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in list(filenames):
            total_checked += 1
            abs_path = os.path.join(dirpath, fname)

            try:
                try:
                    size = os.path.getsize(abs_path)
                except Exception:
                    size = -1

                if size == 0:
                    try:
                        os.remove(abs_path)
                        removed.append((abs_path, "zero_size"))
                    except Exception:
                        removed.append((abs_path, "unremovable_zero"))
                    continue

                safe = _safe_name(fname)
                if safe != fname:
                    new_path = os.path.join(dirpath, safe)
                    base, ext = os.path.splitext(safe)
                    i = 1
                    while os.path.exists(new_path):
                        new_path = os.path.join(dirpath, f"{base}_{i}{ext}")
                        i += 1
                    try:
                        os.rename(abs_path, new_path)
                        renamed.append((abs_path, new_path))
                    except Exception:
                        try:
                            shutil.copy2(abs_path, new_path)
                            os.remove(abs_path)
                            renamed.append((abs_path, new_path))
                        except Exception:
                            pass

            except Exception as e:
                removed.append((abs_path, f"error:{e}"))

    if renamed:
        _emit(socketio, "training_log", {"message": f"🔁 Renamed {len(renamed)} files (normalized)."})
    if removed:
        _emit(socketio, "training_log", {"message": f"🗑️ Removed {len(removed)} problematic files."})

    return {"renamed": renamed, "removed": removed, "checked": total_checked}


# -------------------------
# Progress callback (optional)
# -------------------------
class SocketProgressCallback(Callback):
    def __init__(self, socketio, model_name: str, epochs: int, steps_per_epoch: int):
        super().__init__()
        self.socketio = socketio
        self.model_name = model_name
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch or 1)
        self._epoch_start_time = None

    def on_train_begin(self, logs=None):
        _emit(self.socketio, "training_log", {"message": f"▶️ Start training {self.model_name}"})

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start_time = time.time()
        _emit(self.socketio, "training_log", {"message": f"⏳ Epoch {epoch+1}/{self.epochs} for {self.model_name}"})

    def on_train_batch_end(self, batch, logs=None):
        try:
            percent = int(((batch + 1) / self.steps_per_epoch) * 100)
            _emit(self.socketio, "training_batch", {
                "model": self.model_name,
                "epoch": int(self.params.get("epoch", 0)) + 1,
                "batch": batch + 1,
                "batch_progress": percent,
            })
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        try:
            t = time.time() - (self._epoch_start_time or time.time())
            _emit(self.socketio, "training_epoch", {
                "model": self.model_name,
                "epoch": epoch + 1,
                "epochs": self.epochs,
                "train_acc": float(logs.get("accuracy") * 100) if logs and logs.get("accuracy") is not None else None,
                "val_acc": float(logs.get("val_accuracy") * 100) if logs and logs.get("val_accuracy") is not None else None,
                "epoch_time_s": round(t, 2),
            })
        except Exception:
            pass


# -------------------------
# Dataset
# -------------------------
def make_datasets(
    processed_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    cache: bool,
    seed: int,
    socketio=None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], int]:

    _emit(socketio, "training_log", {"message": "🧹 Dataset normalization... (optional)"})
    normalize_and_clean_dir(processed_dir, socketio=socketio)

    train_dir = os.path.join(processed_dir, "train")
    val_dir = os.path.join(processed_dir, "val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        _emit(socketio, "training_log", {"message": f"📂 using tran/val structure : {processed_dir}"})
        raw_train = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        raw_val = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        _emit(socketio, "training_log", {"message": f"📂 Using internal split  80/20 from: {processed_dir}"})
        raw_train = tf.keras.utils.image_dataset_from_directory(
            processed_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            subset="training",
            seed=seed,
        )
        raw_val = tf.keras.utils.image_dataset_from_directory(
            processed_dir,
            labels="inferred",
            label_mode="categorical",
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            validation_split=0.2,
            subset="validation",
            seed=seed,
        )

    class_names = list(raw_train.class_names)
    steps_per_epoch = int(tf.data.experimental.cardinality(raw_train).numpy())

    def _set_dtype(x, y):
        return tf.cast(x, tf.float32), y

    train_ds = raw_train.map(_set_dtype, num_parallel_calls=AUTOTUNE)
    val_ds = raw_val.map(_set_dtype, num_parallel_calls=AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    _emit(socketio, "training_log", {"message": f"✅ Dataset is ready — {len(class_names)} classes, steps/epoch={steps_per_epoch}."})
    return train_ds, val_ds, class_names, steps_per_epoch


# -------------------------
# Model builders
# -------------------------
def _augmentation_block(seed: int = 123) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, seed=seed),
            layers.RandomZoom(0.10, seed=seed),
        ],
        name="augmentation",
    )


def build_model(backbone_name: str, image_size: Tuple[int, int], num_classes: int, dropout: float, seed: int) -> keras.Model:
    h, w = image_size
    inputs = keras.Input(shape=(h, w, 3), name="image")

    x = layers.Lambda(lambda t: tf.cast(t, tf.float32), name="cast_fp32")(inputs)
    x = _augmentation_block(seed=seed)(x)

    if backbone_name == "DenseNet121":
        preprocess: Callable = tf.keras.applications.densenet.preprocess_input
        base = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    elif backbone_name == "InceptionResNetV2":
        preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input
        base = tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    elif backbone_name == "EfficientNetV2B0":
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
        base = tf.keras.applications.EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    x = layers.Lambda(lambda t: preprocess(t), name="preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=backbone_name)
    model.backbone = base  # attach for easy freezing/unfreezing
    return model


# -------------------------
# Evaluation
# -------------------------
def evaluate_model(model: keras.Model, ds: tf.data.Dataset) -> Dict[str, Any]:
    y_true, y_pred = [], []
    for batch_images, batch_labels in ds:
        preds = model.predict(batch_images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1).tolist())

    if not y_true:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "classification_report": {},
            "confusion_matrix": [],
        }

    cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": float(f_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "classification_report": cr,
        "confusion_matrix": cm,
    }


@dataclass
class TrainConfig:
    processed_dir: str
    models_dir: str
    results_dir: str
    epochs: int = 20
    warmup_epochs: int = 2
    batch_size: int = 16
    image_size: Tuple[int, int] = (224, 224)
    lr: float = 1e-4
    fine_tune_lr: float = 1e-5
    dropout: float = 0.3
    cache_dataset: bool = False
    seed: int = 123


def train_all_cnn(socketio=None, cfg: TrainConfig = None) -> List[Dict[str, Any]]:
    if cfg is None:
        raise ValueError("cfg is required")

    try:
        # mixed precision if GPU available
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy("mixed_float16")
                _emit(socketio, "training_log", {"message": "⚡ GPU detected — mixed precision ON."})
        except Exception:
            pass

        models_cnn_dir = os.path.join(cfg.models_dir, "cnn")
        results_cnn_dir = os.path.join(cfg.results_dir, "cnn")
        os.makedirs(models_cnn_dir, exist_ok=True)
        os.makedirs(results_cnn_dir, exist_ok=True)

        train_ds, val_ds, class_names, steps_per_epoch = make_datasets(
            processed_dir=cfg.processed_dir,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            cache=cfg.cache_dataset,
            seed=cfg.seed,
            socketio=socketio,
        )
        num_classes = len(class_names)
        if num_classes == 0:
            raise RuntimeError("Tidak ada kelas ditemukan di dataset.")

        backbones = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]
        results: List[Dict[str, Any]] = []

        for i, name in enumerate(backbones, start=1):
            _emit(socketio, "training_log", {"message": f"🔧 Preparing model {name} ({i}/{len(backbones)})"})

            model = build_model(
                backbone_name=name,
                image_size=cfg.image_size,
                num_classes=num_classes,
                dropout=cfg.dropout,
                seed=cfg.seed,
            )

            # ---- Stage 1: warmup (freeze backbone)
            model.backbone.trainable = False
            model.compile(
                optimizer=keras.optimizers.Adam(cfg.lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            save_path = os.path.join(models_cnn_dir, f"{name}.keras")
            log_path = os.path.join(results_cnn_dir, f"{name}_train_log.csv")

            callbacks = [
                ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=False),
                CSVLogger(log_path),
                SocketProgressCallback(socketio, name, cfg.epochs, steps_per_epoch),
            ]

            t0 = time.time()

            hist_all: Dict[str, List[float]] = {}
            def _merge_history(h: keras.callbacks.History):
                for k, v in h.history.items():
                    hist_all.setdefault(k, [])
                    hist_all[k].extend([float(x) for x in v])

            warmup = min(cfg.warmup_epochs, cfg.epochs)
            if warmup > 0:
                h1 = model.fit(train_ds, validation_data=val_ds, epochs=warmup, callbacks=callbacks, verbose=0)
                _merge_history(h1)

            # ---- Stage 2: fine-tune (unfreeze backbone)
            remaining = cfg.epochs - warmup
            if remaining > 0:
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
                _merge_history(h2)

            elapsed = time.time() - t0

            # Load BEST checkpoint for evaluation (important!)
            best_model = keras.models.load_model(save_path)

            eval_payload = evaluate_model(best_model, val_ds)
            eval_payload["model"] = name
            eval_payload["num_classes"] = num_classes
            eval_payload["class_names"] = class_names
            eval_payload["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            eval_path = os.path.join(results_cnn_dir, f"{name}_evaluation.json")
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_payload, f, indent=2)

            best_val_acc = float(max(hist_all.get("val_accuracy", [0.0])))

            record = {
                "model": name,
                "accuracy": eval_payload["accuracy"],
                "f1_macro": eval_payload["f1_macro"],
                "precision_macro": eval_payload["precision_macro"],
                "recall_macro": eval_payload["recall_macro"],
                "num_classes": num_classes,
                "class_names": class_names,
                "best_val_accuracy": best_val_acc,
                "train_time_s": round(elapsed, 2),
                "model_path": save_path,
                "evaluation_path": eval_path,
                "history": hist_all,
                "created_at": eval_payload["created_at"],
                "config": asdict(cfg),
            }
            results.append(record)

            _emit(socketio, "training_log", {
                "message": f"✅ Completed {name} — val_best={best_val_acc:.4f} — eval_acc={eval_payload['accuracy']:.4f}"
            })

        # Save ONE full summary (no overwrite)
        summary_path = os.path.join(results_cnn_dir, "training_results_full.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        _emit(socketio, "training_done", {"status": "ok", "results_path": summary_path, "results": results})
        return results

    except Exception as e:
        tb = traceback.format_exc()
        _emit(socketio, "training_error", {"message": str(e), "trace": tb})
        raise


def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True, help="Dataset root (supports train/val/ or class folders).")
    p.add_argument("--models_dir", default="models", help="Where to save models.")
    p.add_argument("--results_dir", default="models/results", help="Where to save JSON results.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--image_size", type=int, default=224, help="Square image size (e.g., 224).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fine_tune_lr", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--cache_dataset", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    return TrainConfig(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        dropout=args.dropout,
        cache_dataset=args.cache_dataset,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    print("Config:", cfg)
    train_all_cnn(socketio=None, cfg=cfg)
    print("Done. Check results in:", os.path.join(cfg.results_dir, "cnn"))
