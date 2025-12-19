# training/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# INPUT DATA (APTOS example)
# =========================================================
# Your real processed data is here:
# G:\My Drive\dr_prototype\processed\aptos2019\processed_final\
#   train\0..4
#   val\0..4
#   test\0..4
PROCESSED_BASE_DIR = r"G:\My Drive\dr_prototype\processed\aptos2019\processed_final"

PROCESSED_TRAIN_DIR = os.path.join(PROCESSED_BASE_DIR, "train")
PROCESSED_VAL_DIR   = os.path.join(PROCESSED_BASE_DIR, "val")
PROCESSED_TEST_DIR  = os.path.join(PROCESSED_BASE_DIR, "test")  # optional, but useful later

# =========================================================
# OUTPUTS
# =========================================================
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_OUT_PATH = os.path.join(MODEL_DIR, "aptos_efficientnetb0.h5")
HISTORY_PLOT_PATH = os.path.join(STATIC_DIR, "training_history.png")

# =========================================================
# DEFAULTS
# =========================================================
NUM_CLASSES = 5
IMG_SIZE = 224
