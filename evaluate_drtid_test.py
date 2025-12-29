# evaluate_drtid_test.py
# ============================================================
# Evaluate DRTiD test set using truth table (testdrtid.csv)
# - Auto-finds latest best_model.keras per backbone under drtid\runs\
# - Predicts on processed test images folder
# - Soft-voting ensemble (avg probs)
# - Computes Accuracy + QWK + Classification report + Confusion matrix
# - Saves CSV + report + confusion matrix image into your ensemble run folder
#
# Includes fixes for:
# - RandomBrightness missing factor during deserialization
# - Unknown layer BackbonePreprocess
# - Any other unknown layers -> identity fallback
# ============================================================

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score


# =========================
# PATHS (YOUR SETUP)
# =========================
RUNS_DIR = Path(r"X:\dr_prototype\processed\drtid\runs")
TEST_DIR = Path(r"X:\dr_prototype\processed\drtid\processed_final\test")
CSV_PATH = Path(r"X:\dr_prototype\dataset\DRTiD\testdrtid.csv")

ENSEMBLE_RUN_DIR = Path(r"X:\dr_prototype\processed\drtid\ensemble_runs\2025-12-28_11-33-12_softvote_by_qwk_val")
OUT_DIR = ENSEMBLE_RUN_DIR / "test_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Because your models include BackbonePreprocess, safest is raw 0..255 float
INPUT_SCALE = "raw"   # "raw" or "01"
BATCH = 32

BACKBONES = ["DenseNet121", "InceptionResNetV2", "EfficientNetV2B0"]


# ============================================================
# Patch: RandomBrightness missing 'factor'
# ============================================================
@tf.keras.utils.register_keras_serializable()
class PatchedRandomBrightness(tf.keras.layers.RandomBrightness):
    def __init__(self, factor=0.0, **kwargs):
        super().__init__(factor=factor, **kwargs)

    @classmethod
    def from_config(cls, config):
        if "factor" not in config:
            for alt in ("brightness_factor", "delta", "max_delta"):
                if alt in config:
                    config["factor"] = config.pop(alt)
                    break
        config.setdefault("factor", 0.0)
        return cls(**config)


# ============================================================
# Custom Layer: BackbonePreprocess
# ============================================================
@tf.keras.utils.register_keras_serializable()
class BackbonePreprocess(tf.keras.layers.Layer):
    def __init__(self, backbone: str = "", **kwargs):
        super().__init__(**kwargs)
        self.backbone = str(backbone or "")

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"backbone": self.backbone})
        return cfg

    @classmethod
    def from_config(cls, config):
        backbone = config.pop("backbone", "")
        if not backbone:
            for k in ("backbone_name", "model_name", "name_backbone", "arch"):
                if k in config:
                    backbone = config.pop(k)
                    break
        return cls(backbone=backbone, **config)

    def call(self, x, training=None):
        x = tf.cast(x, tf.float32)
        x_max = tf.reduce_max(x)
        x01 = tf.less_equal(x_max, 1.5)

        bb = (self.backbone or "").lower()

        def to_255(t):
            return tf.where(x01, t * 255.0, t)

        if "densenet" in bb:
            return tf.keras.applications.densenet.preprocess_input(to_255(x))

        if "inceptionresnet" in bb or "inception_resnet" in bb:
            return tf.keras.applications.inception_resnet_v2.preprocess_input(to_255(x))

        if "efficientnetv2" in bb or "efficientnet_v2" in bb:
            try:
                return tf.keras.applications.efficientnet_v2.preprocess_input(to_255(x))
            except Exception:
                y = to_255(x)
                return (y / 127.5) - 1.0

        return x


# ============================================================
# Identity fallback for unknown layers
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
    custom_objects = dict(base_custom_objects)
    for _ in range(1, 10):
        try:
            return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)
        except ValueError as e:
            msg = str(e)
            m = re.search(r"Unknown layer:\s*([A-Za-z0-9_]+)", msg)
            if not m:
                raise
            unknown = m.group(1)
            if unknown in custom_objects:
                raise
            print(f"⚠️ Unknown layer '{unknown}' while loading. Using identity fallback...")
            custom_objects[unknown] = make_identity_layer_class(unknown)
    raise RuntimeError(f"Failed to load model: {path}")


# ============================================================
# Auto-find latest best_model.keras for a backbone
# ============================================================
def find_latest_best_model(backbone: str) -> Path:
    backbone_l = backbone.lower()
    candidates = []
    for p in RUNS_DIR.rglob("best_model.keras"):
        # check folder name contains backbone
        if backbone_l in str(p.parent).lower():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No best_model.keras found for backbone '{backbone}' under {RUNS_DIR}")

    # newest by modified time
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def infer_input_size(model) -> tuple[int, int]:
    h, w = 224, 224
    try:
        ishape = model.input_shape
        if ishape and len(ishape) >= 4 and ishape[1] and ishape[2]:
            h, w = int(ishape[1]), int(ishape[2])
    except Exception:
        pass
    return h, w


def normalize_key(name: str) -> str:
    # match by filename stem (no extension), lowercase
    return Path(name).stem.lower().strip()


def detect_cols(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}

    id_candidates = ["id_code", "image", "image_id", "filename", "file", "img", "path"]
    y_candidates  = ["diagnosis", "label", "grade", "severity", "class", "dr_grade", "target"]

    id_col = None
    y_col = None

    for k in id_candidates:
        if k in cols_lower:
            id_col = cols_lower[k]
            break

    for k in y_candidates:
        if k in cols_lower:
            y_col = cols_lower[k]
            break

    return id_col, y_col


def main():
    # ---------- sanity checks ----------
    if not RUNS_DIR.exists():
        raise SystemExit(f"❌ RUNS_DIR not found: {RUNS_DIR}")
    if not TEST_DIR.exists():
        raise SystemExit(f"❌ TEST_DIR not found: {TEST_DIR}")
    if not CSV_PATH.exists():
        raise SystemExit(f"❌ CSV not found: {CSV_PATH}")

    # ---------- auto-find models ----------
    model_paths = [find_latest_best_model(bb) for bb in BACKBONES]
    print("✅ Using models:")
    for p in model_paths:
        print(" -", p)

    # ---------- load truth table ----------
    df_truth = pd.read_csv(CSV_PATH)
    id_col, y_col = detect_cols(df_truth)
    if id_col is None or y_col is None:
        raise SystemExit(
            f"❌ Could not detect ID/label columns in CSV.\n"
            f"Columns found: {list(df_truth.columns)}\n"
            f"Rename to something like id_code/filename and diagnosis/label."
        )

    truth_map = {}
    for _, r in df_truth.iterrows():
        key = normalize_key(str(r[id_col]))
        try:
            truth_map[key] = int(r[y_col])
        except Exception:
            continue

    print(f"✅ Truth labels loaded: {len(truth_map)}")
    print(f"   Columns: id={id_col} | label={y_col}")

    # ---------- collect images & match labels ----------
    files_all = list_images(TEST_DIR)
    if not files_all:
        raise SystemExit(f"❌ No images found under: {TEST_DIR}")

    matched_files, y_true = [], []
    missing = 0
    for p in files_all:
        key = normalize_key(p.name)
        if key in truth_map:
            matched_files.append(p)
            y_true.append(truth_map[key])
        else:
            missing += 1

    if len(matched_files) == 0:
        raise SystemExit(
            "❌ No test images matched the CSV IDs.\n"
            "This means CSV IDs and filenames don't align. Share 1 CSV id + 1 filename."
        )

    print(f"✅ Images found: {len(files_all)}")
    print(f"✅ Matched with labels: {len(matched_files)} (unmatched: {missing})")

    # ---------- load models (patched) ----------
    base_custom_objects = {
        "RandomBrightness": PatchedRandomBrightness,
        "BackbonePreprocess": BackbonePreprocess,
    }
    models = [load_model_forgiving(str(p), base_custom_objects) for p in model_paths]

    # ---------- dataset ----------
    img_h, img_w = infer_input_size(models[0])
    IMG_SIZE = (img_h, img_w)
    print(f"✅ Model input size: {IMG_SIZE} | INPUT_SCALE={INPUT_SCALE}")

    def load_image(path_str):
        img_bytes = tf.io.read_file(path_str)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMG_SIZE, antialias=True)
        img = tf.cast(img, tf.float32)
        if INPUT_SCALE == "01":
            img = img / 255.0
        return img

    ds = tf.data.Dataset.from_tensor_slices([str(p) for p in matched_files])
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    # ---------- predict ensemble ----------
    all_probs = []
    for i, m in enumerate(models, start=1):
        probs = m.predict(ds, verbose=1)
        all_probs.append(probs)
        print(f"Model {i} probs shape: {probs.shape}")

    avg_probs = np.mean(all_probs, axis=0)
    y_pred = avg_probs.argmax(axis=1).astype(int)
    conf = avg_probs.max(axis=1)

    # ---------- metrics ----------
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    report = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("\n✅ Accuracy:", round(acc, 6))
    print("✅ QWK:", round(qwk, 6))
    print("\n" + report)

    # ---------- save artifacts ----------
    pred_csv = OUT_DIR / "drtid_test_predictions_with_truth.csv"
    rep_txt  = OUT_DIR / "drtid_test_report.txt"
    cm_png   = OUT_DIR / "drtid_test_confusion_matrix.png"

    out_df = pd.DataFrame({
        "filename": [p.name for p in matched_files],
        "path": [str(p) for p in matched_files],
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": conf,
        "p0": avg_probs[:, 0],
        "p1": avg_probs[:, 1],
        "p2": avg_probs[:, 2],
        "p3": avg_probs[:, 3],
        "p4": avg_probs[:, 4],
    })
    out_df.to_csv(pred_csv, index=False)

    with open(rep_txt, "w", encoding="utf-8") as f:
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"TEST_DIR: {TEST_DIR}\n")
        f.write(f"RUNS_DIR: {RUNS_DIR}\n")
        f.write(f"Models:\n")
        for p in model_paths:
            f.write(f" - {p}\n")
        f.write(f"\nMatched images: {len(matched_files)} / {len(files_all)} (unmatched: {missing})\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"QWK: {qwk:.6f}\n\n")
        f.write(report + "\n")
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))

    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("DRTiD Test Confusion Matrix (Ensemble)")
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, [0,1,2,3,4])
    plt.yticks(tick_marks, [0,1,2,3,4])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(5):
        for j in range(5):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(cm_png, dpi=170)
    plt.close()

    print("\n✅ Saved:")
    print(" -", pred_csv)
    print(" -", rep_txt)
    print(" -", cm_png)


if __name__ == "__main__":
    main()
