# train_models_v2.py
"""
Simpler CLI trainer (ImageDataGenerator) — keeps your original style but fixes:
- num_classes inferred from folder
- correct preprocessing for each backbone (critical)
- uses .keras format (recommended)
"""

import argparse
import json
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(name, input_shape=(224, 224, 3), num_classes=5):
    if name == "densenet121":
        preprocess = tf.keras.applications.densenet.preprocess_input
        base = keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")
    elif name == "inceptionresnetv2":
        preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input
        base = keras.applications.InceptionResNetV2(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")
    elif name == "efficientnetv2b0":
        preprocess = tf.keras.applications.efficientnet_v2.preprocess_input
        base = keras.applications.EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")
    else:
        raise ValueError("Unknown model")

    inputs = keras.Input(shape=input_shape)
    x = layers.Lambda(lambda t: preprocess(t))(inputs)
    x = base(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Must contain train/ and val/ folders.")
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    IMG_SIZE = (224, 224)
    BATCH = args.batch
    EPOCHS = args.epochs

    train_datagen = ImageDataGenerator(
        preprocessing_function=None,  # set per-model via Lambda, so keep None here
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        os.path.join(args.data_dir, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(args.data_dir, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = train_gen.num_classes
    os.makedirs(args.models_dir, exist_ok=True)

    models_to_train = ["densenet121", "inceptionresnetv2", "efficientnetv2b0"]
    summary = []

    for name in models_to_train:
        print("Training", name)
        model = build_model(name, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)
        model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

        ckpt = os.path.join(args.models_dir, f"{name}_best.keras")
        callbacks = [
            ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ]

        start = time.time()
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)
        elapsed = time.time() - start

        val_loss, val_acc = model.evaluate(val_gen)
        summary.append({"model": name, "val_acc": float(val_acc), "val_loss": float(val_loss), "train_time_s": float(elapsed)})

        with open(os.path.join(args.models_dir, f"{name}_history.json"), "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)

    with open(os.path.join(args.models_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training semua model selesai. Summary tersimpan.")
