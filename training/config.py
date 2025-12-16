import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data
PROCESSED_TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "processed_final", "train")

# Outputs
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_OUT_PATH = os.path.join(MODEL_DIR, "aptos_efficientnetb0.h5")
HISTORY_PLOT_PATH = os.path.join(STATIC_DIR, "training_history.png")

# Defaults
NUM_CLASSES = 5
IMG_SIZE = 224
