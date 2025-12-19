# training_cnn_v2.py
"""
Thesis training script (3 CNN backbones): DenseNet121, InceptionResNetV2, EfficientNetV2B0

Key points:
- socketio optional (won't crash if None)
- NO Lambda layers (avoids Keras "unsafe deserialization" errors)
- Robust dataset path handling:
    A) processed_dir/ train/<class>/...   + val/<class>/...
    B) processed_dir is ".../train" and sibling ".../val" exists
    C) processed_dir/<class>/... (fallback 80/20 split)
- Saves BEST and LAST weights for each backbone
- Saves evaluation.json + report.txt + confusion_matrix.png (with numbers)

Outputs per model (inside results_dir/cnn/<ModelName>/):
- best.weights.h5
- last.weights.h5
- best_model.keras
- evaluation.json
- report.txt
- confusion_matrix.png
- train_log.csv
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

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
        pass


# -------------------------
# Class weights
# -------------------------
def compute_class_weights_from_ds(train_ds: tf.data.Dataset, num_classes: int) -> Dict[int, float]:
    """
    Computes sklearn 'balanced' class weights from a tf.data.Dataset where labels are one-hot (categorical).
    Returns dict like {0: w0, 1: w1, ...}
    """
    y_all: List[int] = []
    for _, y in train_ds:
        y_np = y.numpy()
        y_all.extend(np.argmax(y_np, axis=1).tolist())

    y_all = np.array(y_all, dtype=int)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=y_all,
    )
    return {int(i): float(w) for i, w in enumerate(weights)}


# -------------------------
# Filename normalization (optional)
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
    root_dir = str(root_dir)

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
        self.steps_per_epoch = int(max(1, steps_per_epoch or 1))
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
                "batch": int(batch) + 1,
                "batch_progress": percent,
            })
        except Exception:
            pass

    def on_epoch_end(self, epoch, logs=None):
        try:
            t = time.time() - (self._epoch_start_time or time.time())
            _emit(self.socketio, "training_epoch", {
                "model": self.model_name,
                "epoch": int(epoch) + 1,
                "epochs": self.epochs,
                "train_acc": float(logs.get("accuracy") * 100) if logs and logs.get("accuracy") is not None else None,
                "val_acc": float(logs.get("val_accuracy") * 100) if logs and logs.get("val_accuracy") is not None else None,
                "epoch_time_s": round(t, 2),
            })
        except Exception:
            pass


# -------------------------
# Dataset resolver
# -------------------------
def _resolve_train_val_dirs(processed_dir: str) -> Tuple[Optional[str], Optional[str], bool, str]:
    """
    Returns: (train_dir, val_dir, use_internal_split, resolved_root)
    """
    p = os.path.normpath(str(processed_dir))

    # Case A: processed_dir contains train/ and val/
    cand_train = os.path.join(p, "train")
    cand_val = os.path.join(p, "val")
    if os.path.isdir(cand_train) and os.path.isdir(cand_val):
        return cand_train, cand_val, False, p

    # Case B: processed_dir is ".../train" and sibling ".../val" exists
    if os.path.basename(p).lower() == "train":
        parent = os.path.dirname(p)
        sibling_val = os.path.join(parent, "val")
        if os.path.isdir(sibling_val):
            return p, sibling_val, False, parent

    # Case C: fallback internal split from processed_dir/<class>
    return None, None, True, p


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

    train_dir, val_dir, use_internal_split, resolved_root = _resolve_train_val_dirs(processed_dir)

    classes = ["0", "1", "2", "3", "4"]

    if not use_internal_split:
        _emit(socketio, "training_log", {"message": f"📂 Using train/val folders:\n- train: {train_dir}\n- val: {val_dir}"})

        raw_train = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=classes,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        raw_val = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=classes,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        _emit(socketio, "training_log", {"message": f"📂 No val folder found → internal split 80/20 from: {resolved_root}"})

        raw_train = tf.keras.utils.image_dataset_from_directory(
            resolved_root,
            labels="inferred",
            label_mode="categorical",
            class_names=classes,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            validation_split=0.2,
            subset="training",
            seed=seed,
        )
        raw_val = tf.keras.utils.image_dataset_from_directory(
            resolved_root,
            labels="inferred",
            label_mode="categorical",
            class_names=classes,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
            validation_split=0.2,
            subset="validation",
            seed=seed,
        )

    class_names = list(raw_train.class_names)

    # cardinality can be UNKNOWN (-2) → just fallback to 1 for UI progress
    card = int(tf.data.experimental.cardinality(raw_train).numpy())
    steps_per_epoch = card if card > 0 else 1

    def _set_dtype(x, y):
        return tf.cast(x, tf.float32), y

    train_ds = raw_train.map(_set_dtype, num_parallel_calls=AUTOTUNE)
    val_ds = raw_val.map(_set_dtype, num_parallel_calls=AUTOTUNE)

    if cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    _emit(socketio, "training_log", {
        "message": f"✅ Dataset ready — classes={class_names} | steps/epoch≈{steps_per_epoch}"
    })
    return train_ds, val_ds, class_names, steps_per_epoch


# -------------------------
# Model builders (NO Lambda)
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


@keras.utils.register_keras_serializable(package="DR")
class BackbonePreprocessLayer(layers.Layer):
    """
    Serializable preprocessing layer (avoids Lambda).
    """
    def __init__(self, backbone_name: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = str(backbone_name)

    def call(self, x):
        if self.backbone_name == "DenseNet121":
            return tf.keras.applications.densenet.preprocess_input(x)
        if self.backbone_name == "InceptionResNetV2":
            return tf.keras.applications.inception_resnet_v2.preprocess_input(x)
        if self.backbone_name == "EfficientNetV2B0":
            return tf.keras.applications.efficientnet_v2.preprocess_input(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backbone_name": self.backbone_name})
        return cfg


def build_model(
    backbone_name: str,
    image_size: Tuple[int, int],
    num_classes: int,
    dropout: float,
    seed: int,
) -> keras.Model:
    h, w = image_size
    inputs = keras.Input(shape=(h, w, 3), name="image")

    x = _augmentation_block(seed=seed)(inputs)
    x = layers.Rescaling(1.0, offset=0.0, name="identity_rescale")(x)  # keep float32 pipeline
    x = BackbonePreprocessLayer(backbone_name, name="preprocess")(x)

    if backbone_name == "DenseNet121":
        base = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    elif backbone_name == "InceptionResNetV2":
        base = tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    elif backbone_name == "EfficientNetV2B0":
        base = tf.keras.applications.EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(float(dropout), name="dropout")(x)
    outputs = layers.Dense(int(num_classes), activation="softmax", dtype="float32", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=backbone_name)
    model.backbone = base  # attach for freezing/unfreezing
    return model


# -------------------------
# Confusion matrix plot with numbers
# -------------------------
def save_confusion_matrix_png(cm: np.ndarray, class_names: List[str], out_path: str) -> None:
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.int64)
    n = cm.shape[0]

    fig = plt.figure(figsize=(6.8, 6.2))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(n)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True")
    plt.xlabel("Predicted")

    # write numbers
    maxv = cm.max() if cm.size else 0
    thresh = maxv / 2.0 if maxv > 0 else 0
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            plt.text(
                j, i, str(v),
                ha="center", va="center",
                color="white" if v > thresh else "black",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


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
    cm = confusion_matrix(y_true, y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "f1_macro": float(f_macro),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "classification_report": cr,
        "confusion_matrix": cm.tolist(),
        "report_text": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


# -------------------------
# Config
# -------------------------
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
    save_history: bool = False  # default off to keep JSON clean


# -------------------------
# Train all
# -------------------------
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

        # dataset
        train_ds, val_ds, class_names, steps_per_epoch = make_datasets(
            processed_dir=cfg.processed_dir,
            image_size=cfg.image_size,
            batch_size=cfg.batch_size,
            cache=cfg.cache_dataset,
            seed=cfg.seed,
            socketio=socketio,
        )

        num_classes = len(class_names)
        if num_classes <= 0:
            raise RuntimeError("No classes found in dataset.")

        # class weights
        class_weight_dict = compute_class_weights_from_ds(train_ds, num_classes)
        _emit(socketio, "training_log", {"message": f"⚖️ Using class_weight: {class_weight_dict}"})

        backbones = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]
        results: List[Dict[str, Any]] = []

        for i, name in enumerate(backbones, start=1):
            _emit(socketio, "training_log", {"message": f"🔧 Preparing {name} ({i}/{len(backbones)})"})

            # per-model output folder
            out_dir = Path(results_cnn_dir) / name
            out_dir.mkdir(parents=True, exist_ok=True)

            best_w_path = str(out_dir / "best.weights.h5")
            last_w_path = str(out_dir / "last.weights.h5")
            best_model_path = str(out_dir / "best_model.keras")
            eval_json_path = str(out_dir / "evaluation.json")
            report_txt_path = str(out_dir / "report.txt")
            cm_png_path = str(out_dir / "confusion_matrix.png")
            train_log_csv = str(out_dir / "train_log.csv")

            model = build_model(
                backbone_name=name,
                image_size=cfg.image_size,
                num_classes=num_classes,
                dropout=cfg.dropout,
                seed=cfg.seed,
            )

            # Stage 1: warmup (freeze backbone)
            model.backbone.trainable = False
            model.compile(
                optimizer=keras.optimizers.Adam(cfg.lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            callbacks = [
                ModelCheckpoint(
                    filepath=best_w_path,
                    monitor="val_accuracy",
                    save_best_only=True,
                    save_weights_only=True,
                    mode="max",
                    verbose=1,
                ),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
                EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=False),
                CSVLogger(train_log_csv),
                SocketProgressCallback(socketio, name, cfg.epochs, steps_per_epoch),
            ]

            t0 = time.time()
            hist_all: Dict[str, List[float]] = {}

            def _merge_history(h: keras.callbacks.History):
                for k, v in h.history.items():
                    hist_all.setdefault(k, [])
                    hist_all[k].extend([float(x) for x in v])

            warmup = min(int(cfg.warmup_epochs), int(cfg.epochs))

            if warmup > 0:
                h1 = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=warmup,
                    callbacks=callbacks,
                    verbose=0,
                    class_weight=class_weight_dict,
                )
                _merge_history(h1)

            # Stage 2: fine-tune
            remaining = int(cfg.epochs) - warmup
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
                    epochs=int(cfg.epochs),
                    callbacks=callbacks,
                    verbose=0,
                    class_weight=class_weight_dict,
                )
                _merge_history(h2)

            elapsed = time.time() - t0

            # Always save last weights
            model.save_weights(last_w_path)

            # Ensure best exists (fallback to last)
            if not os.path.exists(best_w_path):
                shutil.copy2(last_w_path, best_w_path)

            # Load best weights into model for evaluation
            model.load_weights(best_w_path)

            eval_payload = evaluate_model(model, val_ds)
            eval_payload.update({
                "model": name,
                "num_classes": num_classes,
                "class_names": class_names,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "image_size": list(cfg.image_size),
                "dropout": float(cfg.dropout),
                "best_weights_path": best_w_path,
                "last_weights_path": last_w_path,
                "best_model_path": best_model_path,
                "config": asdict(cfg),
            })

            # save artifacts
            with open(eval_json_path, "w", encoding="utf-8") as f:
                json.dump(eval_payload, f, indent=2)

            Path(report_txt_path).write_text(eval_payload.get("report_text", ""), encoding="utf-8")

            cm_list = eval_payload.get("confusion_matrix") or []
            if cm_list:
                save_confusion_matrix_png(np.array(cm_list, dtype=np.int64), class_names, cm_png_path)

            # Save model (should be safe now because no Lambda + preprocess layer is registered)
            try:
                model.save(best_model_path, include_optimizer=False)
            except Exception:
                # not fatal
                pass

            best_val_acc = float(max(hist_all.get("val_accuracy", [0.0])))

            record = {
                "model": name,
                "accuracy": float(eval_payload["accuracy"]),
                "f1_macro": float(eval_payload["f1_macro"]),
                "precision_macro": float(eval_payload["precision_macro"]),
                "recall_macro": float(eval_payload["recall_macro"]),
                "best_val_accuracy": best_val_acc,
                "train_time_s": round(elapsed, 2),
                "artifacts_dir": str(out_dir),
                "best_weights_path": best_w_path,
                "last_weights_path": last_w_path,
                "best_model_path": best_model_path,
                "evaluation_json": eval_json_path,
                "class_weight": class_weight_dict,
                "created_at": eval_payload["created_at"],
                "config": asdict(cfg),
            }
            if cfg.save_history:
                record["history"] = hist_all

            results.append(record)

            _emit(socketio, "training_log", {
                "message": f"✅ Completed {name} — val_best={best_val_acc:.4f} — eval_acc={eval_payload['accuracy']:.4f}"
            })

        # Save one summary file
        summary_path = os.path.join(results_cnn_dir, "training_results_full.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        _emit(socketio, "training_done", {"status": "ok", "results_path": summary_path, "results": results})
        return results

    except Exception as e:
        tb = traceback.format_exc()
        _emit(socketio, "training_error", {"message": str(e), "trace": tb})
        raise


# -------------------------
# CLI
# -------------------------
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
    p.add_argument("--save_history", action="store_true", help="Include history in summary JSON (bigger file).")
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
        save_history=bool(args.save_history),
    )


if __name__ == "__main__":
    cfg = _parse_args()
    print("Config:", cfg)
    train_all_cnn(socketio=None, cfg=cfg)
    print("Done. Check results in:", os.path.join(cfg.results_dir, "cnn"))
