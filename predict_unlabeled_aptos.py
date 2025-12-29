# predict_unlabeled_aptos.py
# ============================================================
# Predict on UNLABELED, PREPROCESSED APTOS test folder (1928 imgs)
# using 3 saved Keras models (DenseNet121, InceptionResNetV2, EfficientNetV2B0)
# and export ensemble (avg prob) predictions to CSV.
#
# FIXES INCLUDED:
# 1) RandomBrightness deserialization error:
#    "RandomBrightness.__init__() missing 1 required positional argument: 'factor'"
# 2) Unknown custom layer error:
#    "Unknown layer: BackbonePreprocess"
# 3) Extra safety: auto-fallback for any *other* unknown layers (identity placeholder)
# ============================================================

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import tensorflow as tf


# =========================
# YOUR PATHS
# =========================
TEST_DIR = Path(r"X:\dr_prototype\processed\aptos2019\processed_final\test")

MODEL_PATHS = [
    r"X:\dr_prototype\processed\aptos2019\runs\2025-12-28_09-02-41_DenseNet121\best_model.keras",
    r"X:\dr_prototype\processed\aptos2019\runs\2025-12-28_09-30-48_InceptionResNetV2\best_model.keras",
    r"X:\dr_prototype\processed\aptos2019\runs\2025-12-28_09-48-47_EfficientNetV2B0\best_model.keras",
]

OUT_CSV = Path(r"X:\dr_prototype\processed\aptos2019\ensemble_runs\aptos_unlabeled_predictions.csv")


# =========================
# INPUT SCALE (IMPORTANT)
# =========================
# Because your saved models include a custom preprocessing layer (BackbonePreprocess),
# the safest default is to feed RAW pixels (0..255 float32) and let the model preprocess internally.
#
# If your predictions look "wrong" (e.g., almost all one class / very low confidence),
# try changing this to "01".
INPUT_SCALE = "raw"   # "raw" (0..255)  OR  "01" (0..1)


BATCH = 32


# ============================================================
# Patch 1) RandomBrightness missing 'factor'
# ============================================================
@tf.keras.utils.register_keras_serializable()
class PatchedRandomBrightness(tf.keras.layers.RandomBrightness):
    """
    Some TF/Keras versions save RandomBrightness config without 'factor'.
    This patched layer supplies a safe default so load_model() succeeds.
    """
    def __init__(self, factor=0.0, **kwargs):
        super().__init__(factor=factor, **kwargs)

    @classmethod
    def from_config(cls, config):
        if "factor" not in config:
            # possible alternate keys from older versions
            for alt in ("brightness_factor", "delta", "max_delta"):
                if alt in config:
                    config["factor"] = config.pop(alt)
                    break
        config.setdefault("factor", 0.0)
        return cls(**config)


# ============================================================
# Fix 2) Your custom layer: BackbonePreprocess
# ============================================================
@tf.keras.utils.register_keras_serializable()
class BackbonePreprocess(tf.keras.layers.Layer):
    """
    A drop-in implementation to match your saved custom layer name "BackbonePreprocess".
    It applies appropriate keras.applications preprocess_input based on backbone name.

    This is used ONLY so the model can be loaded and used for inference.
    """

    def __init__(self, backbone: str = "", **kwargs):
        super().__init__(**kwargs)
        self.backbone = str(backbone or "")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backbone": self.backbone})
        return cfg

    @classmethod
    def from_config(cls, config):
        # saved configs may use different keys
        backbone = config.pop("backbone", "")
        if not backbone:
            for k in ("backbone_name", "model_name", "name_backbone", "arch"):
                if k in config:
                    backbone = config.pop(k)
                    break
        return cls(backbone=backbone, **config)

    def call(self, x, training=None):
        x = tf.cast(x, tf.float32)

        # Detect whether x is 0..1 or 0..255
        # (we only use this detection inside the layer, in case INPUT_SCALE is changed)
        x_max = tf.reduce_max(x)
        x01 = tf.less_equal(x_max, 1.5)

        bb = (self.backbone or "").lower()

        def to_255(t):
            return tf.where(x01, t * 255.0, t)

        # DenseNet
        if "densenet" in bb:
            return tf.keras.applications.densenet.preprocess_input(to_255(x))

        # Inception-ResNet-V2
        if "inceptionresnet" in bb or "inception_resnet" in bb:
            return tf.keras.applications.inception_resnet_v2.preprocess_input(to_255(x))

        # EfficientNetV2
        if "efficientnetv2" in bb or "efficientnet_v2" in bb:
            # Some TF versions expose efficientnet_v2.preprocess_input, some may not.
            try:
                return tf.keras.applications.efficientnet_v2.preprocess_input(to_255(x))
            except Exception:
                # Fallback: common "tf" style scaling to [-1,1] from 0..255
                y = to_255(x)
                return (y / 127.5) - 1.0

        # Unknown backbone -> identity (no preprocessing)
        return x


# ============================================================
# Extra safety: identity placeholder for any other unknown layers
# (in case a new unknown layer name appears after BackbonePreprocess is fixed)
# ============================================================
def make_identity_layer_class(layer_name: str):
    @tf.keras.utils.register_keras_serializable()
    class _IdentityLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, x, training=None):
            return x

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    _IdentityLayer.__name__ = f"Identity_{re.sub(r'[^0-9a-zA-Z_]+', '_', layer_name)}"
    return _IdentityLayer


def load_model_forgiving(path: str, base_custom_objects: dict):
    """
    Try loading a model; if Keras reports 'Unknown layer: X', inject a placeholder and retry.
    """
    custom_objects = dict(base_custom_objects)

    for attempt in range(1, 8):
        try:
            return tf.keras.models.load_model(
                path,
                compile=False,
                custom_objects=custom_objects,
            )
        except ValueError as e:
            msg = str(e)
            m = re.search(r"Unknown layer:\s*([A-Za-z0-9_]+)", msg)
            if not m:
                raise
            unknown = m.group(1)
            if unknown in custom_objects:
                raise  # already tried; avoid infinite loop
            print(f"⚠️  Unknown layer '{unknown}' while loading. Adding identity fallback and retrying...")
            custom_objects[unknown] = make_identity_layer_class(unknown)

    raise RuntimeError(f"Failed to load model after retries: {path}")


def _list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _infer_input_size(model) -> tuple[int, int]:
    h, w = 224, 224
    try:
        ishape = model.input_shape  # typically (None, H, W, 3)
        if ishape and len(ishape) >= 4 and ishape[1] and ishape[2]:
            h, w = int(ishape[1]), int(ishape[2])
    except Exception:
        pass
    return h, w


def main():
    # =========================
    # 1) Collect images
    # =========================
    files = _list_images(TEST_DIR)
    if not files:
        raise SystemExit(f"❌ No images found in: {TEST_DIR}")
    print(f"✅ Images found: {len(files)}")

    # =========================
    # 2) Load models (patched)
    # =========================
    base_custom_objects = {
        "RandomBrightness": PatchedRandomBrightness,
        "BackbonePreprocess": BackbonePreprocess,
    }

    models = []
    for p in MODEL_PATHS:
        if not Path(p).exists():
            raise SystemExit(f"❌ Model file not found: {p}")
        m = load_model_forgiving(p, base_custom_objects)
        models.append(m)

    print("✅ Loaded models:")
    for p in MODEL_PATHS:
        print(" -", p)

    # =========================
    # 3) Infer input size
    # =========================
    img_h, img_w = _infer_input_size(models[0])
    IMG_SIZE = (img_h, img_w)
    print(f"✅ Model input size: {IMG_SIZE}")
    print(f"✅ INPUT_SCALE: {INPUT_SCALE}")

    # =========================
    # 4) Build dataset
    # =========================
    def load_image(path_str):
        img_bytes = tf.io.read_file(path_str)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE, antialias=True)
        img = tf.cast(img, tf.float32)

        if INPUT_SCALE == "01":
            img = img / 255.0
        # else "raw": keep 0..255 float32

        return img

    paths_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in files])
    ds = (
        paths_ds
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH)
        .prefetch(tf.data.AUTOTUNE)
    )

    # =========================
    # 5) Predict + ensemble (avg prob)
    # =========================
    all_probs = []
    for i, m in enumerate(models, start=1):
        probs = m.predict(ds, verbose=1)
        all_probs.append(probs)
        print(f"Model {i} probs shape: {probs.shape}")

    avg_probs = np.mean(all_probs, axis=0)

    pred_class = avg_probs.argmax(axis=1).astype(int)
    confidence = avg_probs.max(axis=1)

    # =========================
    # 6) Save CSV
    # =========================
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "filename": [p.name for p in files],
        "path": [str(p) for p in files],
        "pred_class": pred_class,
        "confidence": confidence,
        "p0": avg_probs[:, 0],
        "p1": avg_probs[:, 1],
        "p2": avg_probs[:, 2],
        "p3": avg_probs[:, 3],
        "p4": avg_probs[:, 4],
    })
    df.to_csv(OUT_CSV, index=False)

    print("\n✅ Saved:", OUT_CSV)
    print("\nPredicted class counts:")
    print(df["pred_class"].value_counts().sort_index())
    print("\nConfidence summary:")
    print(df["confidence"].describe())


if __name__ == "__main__":
    main()
