# preprocessing.py
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import pandas as pd


# =========================================================
# SocketIO-safe emit helpers (optional)
# =========================================================
def _emit(socketio, event: str, payload: dict):
    """Safe emit (won't crash if socketio is None)."""
    try:
        if socketio is not None:
            socketio.emit(event, payload)
        else:
            # fallback for CLI/debug
            if event.endswith("_error"):
                print(f"[{event}] {payload}")
            elif event.endswith("_log"):
                print(f"[{event}] {payload.get('message','')}")
    except Exception:
        pass


def _pp_log(socketio, msg: str):
    _emit(socketio, "preprocess_log", {"message": msg})


def _pp_err(socketio, msg: str):
    _emit(socketio, "preprocess_error", {"message": msg})


def _pp_progress(socketio, percent: int, processed: int, total: int, phase: str = ""):
    _emit(socketio, "preprocess_progress", {
        "percent": int(percent),
        "processed": int(processed),
        "total": int(total),
        "phase": phase or ""
    })


def _pp_done(socketio, msg: str):
    _emit(socketio, "preprocess_done", {"message": msg})


# =========================================================
# Small utilities (self-contained)
# =========================================================
def _safe_ds(name: str) -> str:
    return "".join(c for c in str(name).strip().lower() if c.isalnum() or c in ("_", "-"))


def _strip_ext(s: str) -> str:
    s = str(s).strip()
    if "." in s:
        return s.rsplit(".", 1)[0]
    return s


def _assign_split(item_id: str, val_ratio: float, seed: int) -> str:
    """
    Deterministic split based on ID string.
    Returns "val" or "train".
    """
    key = f"{seed}:{_strip_ext(item_id)}".encode("utf-8")
    h = hashlib.md5(key).hexdigest()
    r = int(h[:8], 16) / float(0xFFFFFFFF)
    return "val" if r < float(val_ratio) else "train"


def ensure_class_folders(root: Path, num_classes: int = 5):
    root.mkdir(parents=True, exist_ok=True)
    for c in range(num_classes):
        (root / str(c)).mkdir(parents=True, exist_ok=True)


def find_image(img_dir: Path, img_id: str, default_ext: str) -> Optional[str]:
    """
    Try common extensions + default_ext. Returns string path or None.
    """
    img_dir = Path(img_dir)
    base = _strip_ext(img_id)

    exts = []
    if default_ext:
        exts.append(default_ext.lstrip("."))
    exts += ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

    tried = set()
    for e in exts:
        p = img_dir / f"{base}.{e}"
        if str(p).lower() in tried:
            continue
        tried.add(str(p).lower())
        if p.exists():
            return str(p)

    # last resort: try direct name as given
    p2 = img_dir / str(img_id)
    if p2.exists():
        return str(p2)

    return None


# =========================================================
# Fundus mask + robust crop
# =========================================================
def _fundus_mask_and_bbox(gray: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Returns (mask, bbox) where mask is 0/255 and bbox is (x,y,w,h) of largest contour.
    If contour not found, bbox=None and mask is all-ones.
    """
    h, w = gray.shape[:2]
    # Start with a simple threshold like your original, then clean up
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)

    # Morph close to fill small holes
    k = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask, None

    c = max(cnts, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)

    # Build filled contour mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, thickness=-1)

    # If bbox is suspiciously tiny, fallback to full image
    if bw * bh < 0.10 * (w * h):
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return mask, None

    return mask, (x, y, bw, bh)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


# =========================================================
# PREPROCESSING PIPELINE (APTOS + DRTiD)
# - Adaptive CLAHE (clipLimit derived from T/80 idea)
# - LAB L-channel only
# - Mask-based background/ring cleanup
# - Mask-aware gray-world normalization
# =========================================================
def preprocess_image(
    img_bgr: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
    tile_grid: Tuple[int, int] = (8, 8),
    t_div: float = 80.0,
    clip_min: float = 1.0,
    clip_max: float = 4.0,
    sharpen_green: bool = True,
) -> np.ndarray:
    """
    Returns uint8 BGR image resized to out_size.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("empty_image")

    # --- fundus mask + bbox crop (reduces background dominance)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask, bbox = _fundus_mask_and_bbox(gray)

    if bbox is not None:
        x, y, w, h = bbox
        img_bgr = img_bgr[y:y + h, x:x + w]
        mask = mask[y:y + h, x:x + w]

    # Safety: ensure mask matches current image
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = np.ones(img_bgr.shape[:2], dtype=np.uint8) * 255

    m = (mask > 0)

    # --- LAB + adaptive CLAHE on L
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Estimate "global threshold" T from L inside fundus (robust)
    # Using percentile works well and is stable across exposures.
    if m.any():
        T = float(np.percentile(L[m], 90))  # 0..255
    else:
        T = float(np.percentile(L, 90))

    # Paper idea: CLIP LIMIT = T/80 (then clamp to safe range)
    clip = _clamp(T / float(t_div), clip_min, clip_max)

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid)
    L2 = clahe.apply(L)

    img_bgr = cv2.cvtColor(cv2.merge((L2, A, B)), cv2.COLOR_LAB2BGR)

    # --- ring/background cleanup: force outside-fundus to dark
    # (prevents bright outer ring from becoming a learned feature)
    if m.any():
        img_bgr[~m] = 0

    # --- mask-aware gray-world normalization (only inside fundus)
    img = img_bgr.astype(np.float32)
    if m.any():
        pix = img[m].reshape(-1, 3)
        mean = pix.mean(axis=0)  # B,G,R means
        target = float(mean.mean())
        scale = target / (mean + 1e-6)
        img[m] = (pix * scale).reshape(-1, 3)
        img[~m] = 0
    else:
        mean = img.mean(axis=(0, 1))
        img *= float(mean.mean()) / (mean + 1e-6)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # --- optional: green channel sharpening (within fundus only)
    if sharpen_green:
        g = img[:, :, 1].astype(np.float32)
        blur = cv2.GaussianBlur(g, (0, 0), 2)
        g2 = 1.6 * g - 0.6 * blur
        if m.any():
            g[m] = np.clip(g2[m], 0, 255)
        else:
            g = np.clip(g2, 0, 255)
        img[:, :, 1] = g.astype(np.uint8)

    # --- final resize (storage size)
    return cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)


# =========================================================
# Dataset worker (writes train/val/test processed folders)
# =========================================================
def preprocess_dataset_worker(
    dataset_name: str,
    *,
    get_dataset_info,
    get_output_paths,
    val_ratio: float,
    split_seed: int,
    socketio=None,
    limit: Optional[int] = None,
):
    """
    Writes:
      processed_final/train/<class>/*.png
      processed_final/val/<class>/*.png
      processed_final/test/*.png
    """
    try:
        dataset_name = _safe_ds(dataset_name)
        info: Dict[str, Any] = get_dataset_info(dataset_name)
        out: Dict[str, Path] = get_output_paths(dataset_name)

        out_train: Path = out["train"]
        out_val: Path = out["val"]
        out_test: Path = out["test"]

        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
        out_test.mkdir(parents=True, exist_ok=True)

        num_classes = int(info.get("num_classes", 5))
        ensure_class_folders(out_train, num_classes=num_classes)
        ensure_class_folders(out_val, num_classes=num_classes)

        # validate raw paths
        for k in ["train_csv", "train_img_dir", "test_csv", "test_img_dir"]:
            p = Path(info[k])
            if not p.exists():
                _pp_err(socketio, f"❌ [{dataset_name}] Path not found: {p}")
                return

        _pp_log(socketio, f"🟢 Preprocessing started [{dataset_name}]")
        _pp_log(socketio, f"📦 Output base: {out['base']}")
        _pp_log(socketio, f"🔀 Split: train={1.0 - val_ratio:.0%} val={val_ratio:.0%} seed={split_seed}")

        train_df = pd.read_csv(info["train_csv"])
        test_df = pd.read_csv(info["test_csv"])

        id_col = info["id_col"]
        label_col = info["label_col"]
        default_ext = info["ext"]

        if id_col not in train_df.columns or label_col not in train_df.columns:
            _pp_err(socketio, f"❌ [{dataset_name}] train csv missing columns. Need {id_col},{label_col}. Found: {list(train_df.columns)}")
            return

        test_id_col = id_col
        if test_id_col not in test_df.columns:
            candidates = [c for c in ["id_code", "image_id", "image", "filename"] if c in test_df.columns]
            if not candidates:
                _pp_err(socketio, f"❌ [{dataset_name}] test csv missing '{id_col}'. Columns: {list(test_df.columns)}")
                return
            test_id_col = candidates[0]
            _pp_log(socketio, f"⚠️ [{dataset_name}] test csv missing '{id_col}', using '{test_id_col}' instead.")

        if limit is not None and limit > 0:
            train_df = train_df.head(limit)
            test_df = test_df.head(limit)
            _pp_log(socketio, f"🧪 Limit active: train={len(train_df)} test={len(test_df)}")

        total_items = int(len(train_df) + len(test_df))
        done_items = 0

        _pp_progress(socketio, 0, 0, total_items, "starting")
        if socketio is not None:
            socketio.sleep(0)

        # TRAIN/VAL
        processed_train = 0
        processed_val = 0
        skipped_train = 0
        t0 = time.time()

        for _, row in train_df.iterrows():
            img_id = str(row[id_col]).strip()

            try:
                label = int(row[label_col])
                if label < 0 or label >= num_classes:
                    raise ValueError("label_out_of_range")
            except Exception:
                skipped_train += 1
                done_items += 1
                continue

            img_path = find_image(Path(info["train_img_dir"]), img_id, default_ext)
            if not img_path:
                skipped_train += 1
                done_items += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped_train += 1
                done_items += 1
                continue

            try:
                out_img = preprocess_image(img, out_size=(224, 224))
            except Exception:
                skipped_train += 1
                done_items += 1
                continue

            out_name = f"{_strip_ext(img_id)}.png"
            split = _assign_split(img_id, val_ratio=val_ratio, seed=split_seed)

            if split == "val":
                save_path = out_val / str(label) / out_name
                processed_val += 1
                phase = "val"
            else:
                save_path = out_train / str(label) / out_name
                processed_train += 1
                phase = "train"

            cv2.imwrite(str(save_path), out_img)
            done_items += 1

            if done_items % 50 == 0 or done_items == total_items:
                pct = int(round(100.0 * done_items / max(total_items, 1)))
                _pp_progress(socketio, pct, done_items, total_items, phase)
                if socketio is not None:
                    socketio.sleep(0)

            if (processed_train + processed_val) % 300 == 0:
                _pp_log(socketio, f"✅ [{dataset_name}] Train={processed_train} Val={processed_val} | skipped={skipped_train} | {time.time() - t0:.1f}s")

        # TEST
        processed_test = 0
        skipped_test = 0
        t1 = time.time()

        for _, row in test_df.iterrows():
            img_id = str(row[test_id_col]).strip()

            img_path = find_image(Path(info["test_img_dir"]), img_id, default_ext)
            if not img_path:
                skipped_test += 1
                done_items += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped_test += 1
                done_items += 1
                continue

            try:
                out_img = preprocess_image(img, out_size=(224, 224))
            except Exception:
                skipped_test += 1
                done_items += 1
                continue

            out_name = f"{_strip_ext(img_id)}.png"
            save_path = out_test / out_name
            cv2.imwrite(str(save_path), out_img)

            processed_test += 1
            done_items += 1

            if done_items % 50 == 0 or done_items == total_items:
                pct = int(round(100.0 * done_items / max(total_items, 1)))
                _pp_progress(socketio, pct, done_items, total_items, "test")
                if socketio is not None:
                    socketio.sleep(0)

            if processed_test % 300 == 0:
                _pp_log(socketio, f"✅ [{dataset_name}] Test processed: {processed_test} | skipped: {skipped_test} | {time.time() - t1:.1f}s")

        _pp_progress(socketio, 100, total_items, total_items, "done")
        if socketio is not None:
            socketio.sleep(0)

        _pp_done(
            socketio,
            f"🎉 Done [{dataset_name}] | Train: {processed_train} | Val: {processed_val} (skipped {skipped_train}) | "
            f"Test: {processed_test} (skipped {skipped_test}) | Output: {out['base']}"
        )

    except Exception as e:
        _pp_err(socketio, f"❌ [{dataset_name}] Error: {e}")
