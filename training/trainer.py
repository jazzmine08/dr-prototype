import traceback
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

from training.dataset import build_datasets
from training.model import build_model
from training.config import MODEL_OUT_PATH, HISTORY_PLOT_PATH


def run_training(socketio, payload: dict):
    try:
        epochs = int(payload.get("epochs", 10))
        batch_size = int(payload.get("batch_size", 16))
        lr = float(payload.get("lr", 1e-4))
        img_size = int(payload.get("img_size", 224))
        val_split = float(payload.get("val_split", 0.2))

        socketio.emit("training_log", {"message": "🟢 Training started"})
        socketio.emit("training_log", {
            "message": f"Params: epochs={epochs}, batch={batch_size}, lr={lr}"
        })

        train_ds, val_ds, class_names = build_datasets(
            img_size=img_size,
            batch_size=batch_size,
            val_split=val_split
        )

        socketio.emit("training_log", {
            "message": f"Classes detected: {class_names}"
        })

        model = build_model(img_size=img_size, learning_rate=lr)

        class SocketIOCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                socketio.emit("training_log", {
                    "message": (
                        f"Epoch {epoch+1}: "
                        f"loss={logs.get('loss'):.4f}, "
                        f"acc={logs.get('accuracy'):.4f}, "
                        f"val_acc={logs.get('val_accuracy'):.4f}"
                    )
                })
                pct = int((epoch + 1) / epochs * 100)
                socketio.emit("training_progress", {"progress": pct})

        cb = [
            SocketIOCallback(),
            callbacks.ModelCheckpoint(
                MODEL_OUT_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max"
            ),
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=4,
                restore_best_weights=True
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=cb,
            verbose=0
        )

        # Plot history
        plt.figure()
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="val")
        plt.legend()
        plt.title("Training Accuracy")
        plt.savefig(HISTORY_PLOT_PATH)
        plt.close()

        socketio.emit("training_done", {
            "message": "✅ Training completed",
            "model_path": MODEL_OUT_PATH,
            "history_plot": "/static/training_history.png",
            "best_val_acc": max(history.history["val_accuracy"])
        })

    except Exception as e:
        socketio.emit("training_error", {"message": str(e)})
        socketio.emit("training_log", {"message": traceback.format_exc()})
