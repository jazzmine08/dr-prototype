# training/training_cnn_v2.py
# Core CNN utilities (Windows GPU-friendly) for:
#   - DenseNet121
#   - InceptionResNetV2
#   - EfficientNetV2B0
#
# Features:
#   ✅ consistent balancing (balanced_sampling OR class_weight — never both)
#   ✅ stronger augmentation (flip/rot/zoom/shift/contrast/brightness/noise)
#   ✅ deterministic stratified split (optional, can override existing val folder)
#   ✅ writes UI-friendly artifacts:
#        history.json, curves_accuracy.png, curves_loss.png, confusion_matrix.png
#        evaluation.json, report.txt, metrics.json, best_model.keras, best.weights.h5
#
# Folder convention (recommended):
#   X:\dr_prototype\processed\<dataset>\processed_final\
#       train\0..4
#       val\0..4
#       test\0..4 (optional)

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CLASS_NAMES_5 = ["0", "1", "2", "3", "4"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------
# JSON safety
# ---------------------------
def _json_safe(obj: Any) -> Any:
    """Convert numpy/TF/Path objects to JSON-serializable Python types."""
    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    # numpy scalars/arrays
    try:
        import numpy as _np  # noqa
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # tensorflow tensors
    try:
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
    except Exception:
        pass

    if isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _write_json(p: Path, payload: Dict[str, Any]):
    p.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _emit(socketio, event: str, payload: dict):
    """Safe emit (won't crash if socketio is None)."""
    try:
        if socketio is not None:
            socketio.emit(event, payload)
        else:
            print(f"[{event}] {payload}")
    except Exception:
        pass


# ---------------------------
# Config
# ---------------------------
@dataclass
class TrainConfig:
    dataset: str
    processed_dir: Path  # .../processed_final/train OR .../processed_final

    epochs: int = 6
    warmup_epochs: int = 1
    batch_size: int = 16
    img_size: int = 224
    lr: float = 1e-4
    fine_tune_lr: float = 1e-5
    dropout: float = 0.3
    seed: int = 123
    cache_dataset: bool = False

    # balancing
    balanced_sampling: bool = False  # if True -> balanced sampler; class_weight disabled

    # augmentation
    use_augmentation: bool = True
    aug_flip: str = "horizontal"  # "horizontal" | "horizontal_and_vertical" | "none"
    aug_rot_deg: float = 30.0
    aug_zoom: float = 0.20
    aug_shift: float = 0.05
    aug_contrast: float = 0.20
    aug_brightness: float = 0.15
    aug_noise_std: float = 0.00

    # stratified split
    force_stratified_split: bool = False
    val_ratio: float = 0.20

    save_history: bool = True


# ---------------------------
# Custom layers (must be loadable for best_model.keras)
# ---------------------------
class RandomBrightness(keras.layers.Layer):
    def __init__(self, max_delta: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.max_delta = float(max_delta)

    def call(self, x, training=None):
        if training and self.max_delta > 0:
            return tf.image.random_brightness(x, max_delta=self.max_delta)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_delta": self.max_delta})
        return cfg


class BackbonePreprocess(keras.layers.Layer):
    def __init__(self, backbone_name: str, **kwargs):
        super().__init__(**kwargs)
        self.backbone_name = str(backbone_name)

    def call(self, x, training=None):
        b = self.backbone_name.lower()
        if "densenet" in b:
            return tf.keras.applications.densenet.preprocess_input(x)
        if "inceptionresnet" in b or "inception_resnet" in b:
            return tf.keras.applications.inception_resnet_v2.preprocess_input(x)
        if "efficientnetv2" in b or "efficientnet_v2" in b:
            return tf.keras.applications.efficientnet_v2.preprocess_input(x)
        return (tf.cast(x, tf.float32) / 127.5) - 1.0

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backbone_name": self.backbone_name})
        return cfg


# ---------------------------
# Augmentation
# ---------------------------
def _augmentation_block(cfg: TrainConfig) -> keras.Sequential:
    flip = (cfg.aug_flip or "horizontal").strip().lower()
    if flip not in ("horizontal", "horizontal_and_vertical", "none"):
        flip = "horizontal"

    rot_factor = float(cfg.aug_rot_deg or 0) / 360.0
    zoom = float(cfg.aug_zoom or 0)
    shift = float(cfg.aug_shift or 0)

    layers: List[keras.layers.Layer] = []
    if flip != "none":
        layers.append(keras.layers.RandomFlip(flip))
    if rot_factor > 0:
        layers.append(keras.layers.RandomRotation(rot_factor))
    if zoom > 0:
        layers.append(keras.layers.RandomZoom(height_factor=(-zoom, zoom), width_factor=(-zoom, zoom)))
    if shift > 0:
        layers.append(keras.layers.RandomTranslation(height_factor=(-shift, shift), width_factor=(-shift, shift)))
    if float(cfg.aug_contrast or 0) > 0:
        layers.append(keras.layers.RandomContrast(float(cfg.aug_contrast)))
    if float(cfg.aug_brightness or 0) > 0:
        layers.append(RandomBrightness(max_delta=float(cfg.aug_brightness)))
    if float(cfg.aug_noise_std or 0) > 0:
        layers.append(keras.layers.GaussianNoise(stddev=float(cfg.aug_noise_std)))

    return keras.Sequential(layers, name="augmentation")


# ---------------------------
# Model builder
# ---------------------------
def build_model(
    backbone_name: str,
    image_size: Tuple[int, int],
    num_classes: int,
    cfg: TrainConfig,
) -> Tuple[keras.Model, keras.Model]:
    """Returns (full_model, backbone_model)."""
    tf.keras.utils.set_random_seed(int(cfg.seed))

    h, w = int(image_size[0]), int(image_size[1])
    inp = keras.Input(shape=(h, w, 3), name="image")

    x = inp
    if cfg.use_augmentation:
        x = _augmentation_block(cfg)(x)

    x = BackbonePreprocess(backbone_name, name="preprocess")(x)

    b = (backbone_name or "").lower()
    if "densenet" in b:
        base = keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(h, w, 3), pooling="avg")
    elif "inceptionresnet" in b or "inception_resnet" in b:
        base = keras.applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(h, w, 3), pooling="avg")
    elif "efficientnetv2" in b or "efficientnet_v2" in b:
        base = keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(h, w, 3), pooling="avg")
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base.trainable = False
    feat = base(x, training=False)

    head = keras.layers.Dropout(float(cfg.dropout))(feat)
    out = keras.layers.Dense(int(num_classes), activation="softmax", name="pred")(head)

    model = keras.Model(inp, out, name=f"{backbone_name}_classifier")
    return model, base


def _freeze_batchnorm(base: keras.Model):
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False


# ---------------------------
# Dataset helpers
# ---------------------------
def _infer_dirs(processed_dir: Path) -> Tuple[Path, Path, Path]:
    p = Path(processed_dir).resolve()
    if p.name.lower() == "train":
        root = p.parent
        return root, p, root / "val"
    if p.name.lower() == "processed_final":
        root = p
        return root, root / "train", root / "val"
    root = p.parent
    return root, p, root / "val"


def _list_images_in_dir(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out: List[Path] = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def _load_image_tf(path: tf.Tensor, image_size: Tuple[int, int]) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, image_size, method="bilinear", antialias=True)
    return img * 255.0  # [0,255]


def _ds_from_filepaths(
    filepaths: List[str],
    labels: np.ndarray,
    image_size: Tuple[int, int],
    batch_size: int,
    num_classes: int,
    shuffle: bool,
    seed: int,
    cache: bool,
) -> tf.data.Dataset:
    fp = tf.constant(filepaths)
    y = tf.constant(labels.astype(np.int32))
    ds = tf.data.Dataset.from_tensor_slices((fp, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 10000), seed=seed, reshuffle_each_iteration=True)

    def _map(path, lab):
        img = _load_image_tf(path, image_size=image_size)
        oh = tf.one_hot(lab, depth=num_classes)
        return img, oh

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def _gather_pool_files(train_dir: Path, val_dir: Optional[Path]) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    fps: List[str] = []
    ys: List[int] = []
    counts: Dict[str, int] = {}

    for cls_idx, cls_name in enumerate(CLASS_NAMES_5):
        cls_count = 0
        for base in [train_dir, val_dir] if val_dir is not None else [train_dir]:
            if base is None:
                continue
            d = base / cls_name
            files = _list_images_in_dir(d)
            for f in files:
                fps.append(str(f))
                ys.append(cls_idx)
            cls_count += len(files)
        counts[cls_name] = int(cls_count)

    return fps, np.asarray(ys, dtype=np.int32), counts


def _stratified_split(
    filepaths: List[str],
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    vr = float(val_ratio)
    if not (0.05 <= vr <= 0.5):
        vr = 0.2
    sss = StratifiedShuffleSplit(n_splits=1, test_size=vr, random_state=int(seed))
    idx_train, idx_val = next(sss.split(np.zeros(len(labels)), labels))
    train_fp = [filepaths[i] for i in idx_train]
    val_fp = [filepaths[i] for i in idx_val]
    return (train_fp, labels[idx_train]), (val_fp, labels[idx_val])


def _compute_class_weight_from_counts(counts: Dict[str, int]) -> Dict[int, float]:
    total = float(sum(int(v) for v in counts.values()))
    present = [k for k, v in counts.items() if int(v) > 0]
    k = float(len(present))
    if total <= 0 or k <= 0:
        return {}
    out: Dict[int, float] = {}
    for i, cls in enumerate(CLASS_NAMES_5):
        n = float(int(counts.get(cls, 0)))
        if n <= 0:
            continue
        out[i] = total / (k * n)
    return out


def _make_balanced_train_ds(
    train_ds: tf.data.Dataset,
    class_counts: Dict[str, int],
    batch_size: int,
    seed: int,
) -> Tuple[tf.data.Dataset, Dict[str, Any], int]:
    present = [i for i, c in enumerate(CLASS_NAMES_5) if int(class_counts.get(c, 0)) > 0]
    if not present:
        return train_ds, {"note": "no classes found"}, 0

    target = max(int(class_counts.get(CLASS_NAMES_5[i], 0)) for i in present)
    target = max(1, int(target))

    base = train_ds.unbatch().cache()
    per_class = []
    for c in present:
        ds_c = base.filter(lambda x, y: tf.equal(tf.argmax(y, axis=-1, output_type=tf.int32), tf.constant(c, tf.int32)))
        ds_c = ds_c.shuffle(2048, seed=seed, reshuffle_each_iteration=True).repeat()
        per_class.append(ds_c)

    mixed = tf.data.Dataset.sample_from_datasets(per_class, weights=[1.0] * len(per_class), seed=seed)
    total_per_epoch = target * len(present)
    steps = int(math.ceil(total_per_epoch / float(batch_size)))

    mixed = mixed.take(total_per_epoch).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    info = {
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "present_classes": present,
        "target_per_class_per_epoch": int(target),
        "steps_per_epoch": int(steps),
        "total_per_epoch": int(total_per_epoch),
    }
    return mixed, info, steps


def make_datasets(
    processed_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    cache: bool,
    seed: int,
    socketio=None,
    balanced_sampling: bool = False,
    force_stratified_split: bool = False,
    val_ratio: float = 0.2,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str], int, Dict[str, Any]]:
    root, train_dir, val_dir = _infer_dirs(Path(processed_dir))
    class_names = list(CLASS_NAMES_5)

    val_exists = val_dir.exists() and any((val_dir / c).exists() for c in class_names)
    use_folder_val = val_exists and (not force_stratified_split)

    extra: Dict[str, Any] = {
        "root": str(root),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "used_folder_val": bool(use_folder_val),
        "force_stratified_split": bool(force_stratified_split),
        "val_ratio": float(val_ratio),
    }

    if use_folder_val:
        _emit(socketio, "train_log", {"message": f"📂 Using train/val folders: train={train_dir} | val={val_dir}"})
        train_ds = tf.keras.utils.image_dataset_from_directory(
            str(train_dir),
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            str(val_dir),
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        )

        # ensure float32 for preprocess (expects 0..255 float)
        def _cast(x, y):
            return tf.cast(x, tf.float32), y

        train_ds = train_ds.map(_cast, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(_cast, num_parallel_calls=tf.data.AUTOTUNE)

        if cache:
            train_ds = train_ds.cache()
            val_ds = val_ds.cache()
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        class_counts = {c: len(_list_images_in_dir(train_dir / c)) for c in class_names}
        steps_per_epoch = int(math.ceil(sum(class_counts.values()) / float(batch_size)))
        extra["class_counts_train"] = {k: int(v) for k, v in class_counts.items()}
        extra["steps_per_epoch"] = int(steps_per_epoch)
    else:
        _emit(socketio, "train_log", {"message": "🧪 Using STRATIFIED split (in-memory) — ignoring existing val folder."})
        fps, y, counts_pool = _gather_pool_files(train_dir, val_dir if val_dir.exists() else None)
        if len(fps) == 0:
            raise RuntimeError(f"No images found under: {train_dir} (val={val_dir})")

        (train_fp, y_train), (val_fp, y_val) = _stratified_split(fps, y, val_ratio=val_ratio, seed=seed)
        train_ds = _ds_from_filepaths(
            train_fp, y_train, image_size=image_size, batch_size=batch_size,
            num_classes=len(class_names), shuffle=True, seed=seed, cache=cache
        )
        val_ds = _ds_from_filepaths(
            val_fp, y_val, image_size=image_size, batch_size=batch_size,
            num_classes=len(class_names), shuffle=False, seed=seed, cache=cache
        )

        steps_per_epoch = int(math.ceil(len(train_fp) / float(batch_size)))
        class_counts = {c: int(np.sum(y_train == i)) for i, c in enumerate(class_names)}

        extra["class_counts_pool"] = {k: int(v) for k, v in counts_pool.items()}
        extra["num_pool"] = int(len(fps))
        extra["num_train"] = int(len(train_fp))
        extra["num_val"] = int(len(val_fp))
        extra["class_counts_train"] = {k: int(v) for k, v in class_counts.items()}
        extra["steps_per_epoch"] = int(steps_per_epoch)

    if balanced_sampling:
        balanced_train, bal_info, steps_bal = _make_balanced_train_ds(
            train_ds=train_ds,
            class_counts=extra.get("class_counts_train", {}),
            batch_size=batch_size,
            seed=seed,
        )
        train_ds = balanced_train
        steps_per_epoch = int(steps_bal)
        extra["balanced_info"] = bal_info
        extra["steps_per_epoch"] = int(steps_per_epoch)
        _emit(socketio, "train_log", {"message": f"⚖️ Balanced sampling ON | steps/epoch={steps_per_epoch} | target/class={bal_info.get('target_per_class_per_epoch')}"})
    else:
        extra["balanced_info"] = None

    return train_ds, val_ds, class_names, int(steps_per_epoch), extra


# ---------------------------
# Evaluation
# ---------------------------
def _y_to_int(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(int)
    return y.astype(int)


def evaluate_model(model: keras.Model, val_ds: tf.data.Dataset, class_names: List[str]) -> Dict[str, Any]:
    y_true: List[int] = []
    y_pred: List[int] = []
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        pred = np.argmax(probs, axis=1).astype(int)
        y_np = yb.numpy() if hasattr(yb, "numpy") else np.asarray(yb)
        true = _y_to_int(y_np)
        y_true.extend(true.tolist())
        y_pred.extend(pred.tolist())

    acc = float(accuracy_score(y_true, y_pred))
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    pm = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rm = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    return {
        "accuracy": acc,
        "kappa_qwk": qwk,
        "f1_macro": f1m,
        "precision_macro": pm,
        "recall_macro": rm,
        "report": report,
        "confusion_matrix": cm,
        "num_samples": int(len(y_true)),
    }


# ---------------------------
# Plots
# ---------------------------
def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, title: str = "Confusion Matrix"):
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
                fontsize=11,
            )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=170)
    plt.close(fig)


def _plot_curves(history_all: Dict[str, Any], out_dir: Path):
    """Save curves_accuracy.png and curves_loss.png (if keys exist)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _concat(key: str) -> Optional[List[float]]:
        vals: List[float] = []
        for part in ("warmup", "finetune"):
            h = history_all.get(part) or {}
            if isinstance(h, dict) and key in h and isinstance(h[key], list):
                vals.extend([float(x) for x in h[key]])
        return vals if vals else None

    acc_train = _concat("accuracy") or _concat("categorical_accuracy")
    acc_val = _concat("val_accuracy") or _concat("val_categorical_accuracy")
    if acc_train or acc_val:
        fig = plt.figure()
        if acc_train:
            plt.plot(acc_train)
        if acc_val:
            plt.plot(acc_val)
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        labels = []
        if acc_train:
            labels.append("train")
        if acc_val:
            labels.append("val")
        if labels:
            plt.legend(labels, loc="best")
        plt.tight_layout()
        plt.savefig(str(out_dir / "curves_accuracy.png"), dpi=150)
        plt.close(fig)

    loss_train = _concat("loss")
    loss_val = _concat("val_loss")
    if loss_train or loss_val:
        fig = plt.figure()
        if loss_train:
            plt.plot(loss_train)
        if loss_val:
            plt.plot(loss_val)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        labels = []
        if loss_train:
            labels.append("train")
        if loss_val:
            labels.append("val")
        if labels:
            plt.legend(labels, loc="best")
        plt.tight_layout()
        plt.savefig(str(out_dir / "curves_loss.png"), dpi=150)
        plt.close(fig)


def _best_from_history(history_all: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """Returns (best_val_accuracy, best_val_loss) across warmup+finetune."""
    vals_acc: List[float] = []
    vals_loss: List[float] = []

    for part in ("warmup", "finetune"):
        h = history_all.get(part) or {}
        if not isinstance(h, dict):
            continue

        if "val_accuracy" in h and isinstance(h["val_accuracy"], list):
            vals_acc.extend([float(x) for x in h["val_accuracy"]])
        elif "val_categorical_accuracy" in h and isinstance(h["val_categorical_accuracy"], list):
            vals_acc.extend([float(x) for x in h["val_categorical_accuracy"]])

        if "val_loss" in h and isinstance(h["val_loss"], list):
            vals_loss.extend([float(x) for x in h["val_loss"]])

    best_val_acc = max(vals_acc) if vals_acc else None
    best_val_loss = min(vals_loss) if vals_loss else None
    return best_val_acc, best_val_loss


# ---------------------------
# SocketIO progress callback (brings back epoch progress in your log)
# ---------------------------
class SocketEpochLogger(keras.callbacks.Callback):
    def __init__(self, socketio=None, prefix: str = ""):
        super().__init__()
        self.socketio = socketio
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = (
            f"{self.prefix}Epoch {epoch+1}/{self.params.get('epochs', '?')} | "
            f"loss={float(logs.get('loss', 0.0)):.4f} | "
            f"acc={float(logs.get('accuracy', logs.get('categorical_accuracy', 0.0))):.4f} | "
            f"val_loss={float(logs.get('val_loss', 0.0)):.4f} | "
            f"val_acc={float(logs.get('val_accuracy', logs.get('val_categorical_accuracy', 0.0))):.4f}"
        )
        _emit(self.socketio, "train_log", {"message": msg})


# ---------------------------
# Training
# ---------------------------
def train_one_backbone(
    cfg: TrainConfig,
    backbone_name: str,
    run_dir: Path,
    socketio=None,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    image_size = (int(cfg.img_size), int(cfg.img_size))
    num_classes = len(CLASS_NAMES_5)

    _emit(socketio, "train_log", {"message": f"🏗️ Building datasets | balanced_sampling={cfg.balanced_sampling} | stratified={cfg.force_stratified_split}"})
    train_ds, val_ds, class_names, steps_per_epoch, ds_info = make_datasets(
        processed_dir=str(cfg.processed_dir),
        image_size=image_size,
        batch_size=int(cfg.batch_size),
        cache=bool(cfg.cache_dataset),
        seed=int(cfg.seed),
        socketio=socketio,
        balanced_sampling=bool(cfg.balanced_sampling),
        force_stratified_split=bool(cfg.force_stratified_split),
        val_ratio=float(cfg.val_ratio),
    )

    _emit(socketio, "train_log", {"message": f"✅ Dataset ready | classes={class_names} | steps/epoch≈{steps_per_epoch}"})

    class_weight = None
    if not cfg.balanced_sampling:
        cw = _compute_class_weight_from_counts(ds_info.get("class_counts_train", {}))
        class_weight = cw if cw else None
        if class_weight:
            _emit(socketio, "train_log", {"message": f"⚖️ class_weight ON (balanced_sampling OFF): {class_weight}"})

    _emit(socketio, "train_log", {"message": f"🧠 Building model: {backbone_name}"})
    model, base = build_model(backbone_name=backbone_name, image_size=image_size, num_classes=num_classes, cfg=cfg)

    best_w = run_dir / "best.weights.h5"
    last_w = run_dir / "last.weights.h5"

    cb = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_w),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=0,
        ),
        SocketEpochLogger(socketio=socketio, prefix=f"{backbone_name} | "),
    ]

    history_all: Dict[str, Any] = {"warmup": None, "finetune": None}
    warm = int(max(0, cfg.warmup_epochs))
    total = int(max(1, cfg.epochs))

    if warm > 0:
        _emit(socketio, "train_log", {"message": f"🔥 Warmup epochs={warm} lr={cfg.lr}"})
        base.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(float(cfg.lr)), loss="categorical_crossentropy", metrics=["accuracy"])
        h1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=warm,
            steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
            class_weight=class_weight,
            callbacks=cb,
            verbose=0,
        )
        history_all["warmup"] = h1.history

    finetune_epochs = max(0, total - warm)
    _emit(socketio, "train_log", {"message": f"🔧 Fine-tune epochs={finetune_epochs} lr={cfg.fine_tune_lr}"})
    base.trainable = True
    _freeze_batchnorm(base)

    model.compile(optimizer=keras.optimizers.Adam(float(cfg.fine_tune_lr)), loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=warm,
        epochs=total,
        steps_per_epoch=steps_per_epoch if steps_per_epoch > 0 else None,
        class_weight=class_weight,
        callbacks=cb,
        verbose=0,
    )
    history_all["finetune"] = h2.history

    # save history for UI/debug (JSON-safe)
    if cfg.save_history:
        _write_json(run_dir / "history.json", history_all)

    # save curves (Results page images)
    _plot_curves(history_all, run_dir)

    # save weights
    model.save_weights(str(last_w))
    if best_w.exists():
        model.load_weights(str(best_w))

    # save full model for robust ensemble loading
    model_path = run_dir / "best_model.keras"
    try:
        model.save(str(model_path), include_optimizer=False)
    except Exception:
        model_path = None

    # evaluation
    eval_res = evaluate_model(model, val_ds, class_names=class_names)
    _write_json(run_dir / "evaluation.json", eval_res)
    (run_dir / "report.txt").write_text(eval_res.get("report", ""), encoding="utf-8")

    # confusion matrix image
    try:
        _plot_confusion_matrix(
            cm=np.asarray(eval_res.get("confusion_matrix", []), dtype=np.int64),
            class_names=list(class_names),
            out_path=(run_dir / "confusion_matrix.png"),
            title="Confusion Matrix",
        )
    except Exception as e:
        _emit(socketio, "train_log", {"message": f"⚠️ Failed to save confusion_matrix.png: {e}"})

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    best_val_acc, best_val_loss = _best_from_history(history_all)

    # UI-friendly metrics.json (top-level keys + summary/extra for compatibility)
    metrics = {
        "dataset": cfg.dataset,
        "model": backbone_name,
        "backbone": backbone_name,
        "created_at": created_at,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),

        "kappa_qwk": float(eval_res["kappa_qwk"]),
        "best_val_accuracy": float(best_val_acc) if best_val_acc is not None else float(eval_res["accuracy"]),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "num_val_samples": int(eval_res["num_samples"]),

        "artifacts": {
            "curves_accuracy.png": str(run_dir / "curves_accuracy.png"),
            "curves_loss.png": str(run_dir / "curves_loss.png"),
            "confusion_matrix.png": str(run_dir / "confusion_matrix.png"),
            "report.txt": str(run_dir / "report.txt"),
            "evaluation.json": str(run_dir / "evaluation.json"),
            "history.json": str(run_dir / "history.json"),
            "best.weights.h5": str(best_w),
            "last.weights.h5": str(last_w),
            "best_model.keras": str(model_path) if model_path else "",
        },

        "extra": {
            "image_size": [int(cfg.img_size), int(cfg.img_size)],
            "batch_size": int(cfg.batch_size),
            "epochs": int(cfg.epochs),
            "warmup_epochs": int(cfg.warmup_epochs),
            "lr": float(cfg.lr),
            "fine_tune_lr": float(cfg.fine_tune_lr),
            "dropout": float(cfg.dropout),
            "seed": int(cfg.seed),
            "balanced_sampling": bool(cfg.balanced_sampling),
            "balanced_info": ds_info.get("balanced_info"),
            "class_counts_train": ds_info.get("class_counts_train"),
            "use_augmentation": bool(cfg.use_augmentation),
            "aug_flip": cfg.aug_flip,
            "aug_rot_deg": float(cfg.aug_rot_deg),
            "aug_zoom": float(cfg.aug_zoom),
            "aug_shift": float(cfg.aug_shift),
            "aug_contrast": float(cfg.aug_contrast),
            "aug_brightness": float(cfg.aug_brightness),
            "aug_noise_std": float(cfg.aug_noise_std),
            "force_stratified_split": bool(cfg.force_stratified_split),
            "val_ratio": float(cfg.val_ratio),
            "config": asdict(cfg),
        },

        "summary": {
            "dataset": cfg.dataset,
            "model": backbone_name,
            "backbone": backbone_name,
            "created_at": created_at,
            "best_val_accuracy": float(best_val_acc) if best_val_acc is not None else float(eval_res["accuracy"]),
            "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
            "kappa_qwk": float(eval_res["kappa_qwk"]),
            "num_samples": int(eval_res["num_samples"]),
        },
    }
    _write_json(run_dir / "metrics.json", metrics)

    return {
        "dataset": cfg.dataset,
        "model": backbone_name,
        "accuracy": float(eval_res["accuracy"]),
        "f1_macro": float(eval_res["f1_macro"]),
        "precision_macro": float(eval_res["precision_macro"]),
        "recall_macro": float(eval_res["recall_macro"]),
        "kappa_qwk": float(eval_res["kappa_qwk"]),
        "best_val_accuracy": metrics["best_val_accuracy"],
        "best_val_loss": metrics["best_val_loss"],
        "num_val_samples": int(eval_res["num_samples"]),
        "best_weights_path": str(best_w),
        "last_weights_path": str(last_w),
        "best_model_path": str(model_path) if model_path else "",
        "evaluation_path": str(run_dir / "evaluation.json"),
        "run_dir": str(run_dir),
        "created_at": created_at,
        "balanced_sampling": bool(cfg.balanced_sampling),
        "balanced_info": ds_info.get("balanced_info"),
        "use_augmentation": bool(cfg.use_augmentation),
        "force_stratified_split": bool(cfg.force_stratified_split),
        "val_ratio": float(cfg.val_ratio),
        "config": asdict(cfg),
    }
