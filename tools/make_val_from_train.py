import random, shutil
from pathlib import Path

random.seed(42)

BASE = Path(r"G:\My Drive\dr_prototype\processed\aptos2019\processed_final")
TRAIN_DIR = BASE / "train"
VAL_DIR   = BASE / "val"

VAL_FRAC = 0.2  # 20% of each class to validation

# delete old val if exists
if VAL_DIR.exists():
    shutil.rmtree(VAL_DIR)

for c in ["0","1","2","3","4"]:
    src = TRAIN_DIR / c
    dst = VAL_DIR / c
    dst.mkdir(parents=True, exist_ok=True)

    files = list(src.glob("*.*"))
    if len(files) == 0:
        raise RuntimeError(f"Empty class folder: {src}")

    random.shuffle(files)
    n_val = max(1, int(len(files) * VAL_FRAC))

    for f in files[:n_val]:
        shutil.move(str(f), str(dst / f.name))  # move from train -> val

    print(f"Class {c}: moved {n_val} to val, remaining {len(files)-n_val} in train")

print("DONE. Validation created at:", VAL_DIR)
