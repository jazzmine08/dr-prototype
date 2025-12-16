import tensorflow as tf
from training.config import PROCESSED_TRAIN_DIR

def build_datasets(
    img_size=224,
    batch_size=16,
    val_split=0.2,
    seed=42
):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="validation"
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, train_ds.class_names
