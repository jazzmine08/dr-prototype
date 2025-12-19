# training/dataset.py
import tensorflow as tf
from training.config import PROCESSED_TRAIN_DIR, PROCESSED_VAL_DIR

def build_datasets(
    img_size=224,
    batch_size=16,
    seed=42
):
    """
    Loads datasets from two separate folders (no validation_split):
      - PROCESSED_TRAIN_DIR = .../processed_final/train
      - PROCESSED_VAL_DIR   = .../processed_final/val

    This fixes the issue where class 0 and 1 can disappear from validation when using
    image_dataset_from_directory(validation_split=..., subset="validation") which is not stratified.
    """
    classes = ["0", "1", "2", "3", "4"]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        class_names=classes,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_VAL_DIR,
        labels="inferred",
        label_mode="int",
        class_names=classes,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,  # keep validation deterministic
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, classes
