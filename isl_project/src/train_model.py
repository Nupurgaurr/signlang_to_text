

import argparse
import sys
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils import (
    get_logger, MODEL_DIR, PROCESSED_DIR, LABEL_MAP_PATH,
    IMG_SIZE, LANDMARK_DIM, load_label_map, MODEL_H5_PATH, LANDMARK_H5_PATH,
)
from preprocessing import extract_landmarks_from_dataset

log = get_logger("train_model")

EPOCHS_LM  = 80
EPOCHS_CNN = 30
BATCH_SIZE = 64


# ── Landmark Model ─────────────────────────────────────────────────────────────
def build_landmark_model(num_classes: int) -> keras.Model:
    inp = keras.Input(shape=(LANDMARK_DIM,), name="landmarks")
    x = layers.Dense(512, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = keras.Model(inp, out, name="landmark_model")
    return model


def train_landmark_model():
    log.info("═══ Training Landmark Model ═══")

    X_tr = PROCESSED_DIR / "train_landmarks.npy"
    Y_tr = PROCESSED_DIR / "train_labels.npy"

    if not X_tr.exists():
        log.info("Landmark features not found — extracting …")
        extract_landmarks_from_dataset("train")
    if not (PROCESSED_DIR / "test_landmarks.npy").exists():
        extract_landmarks_from_dataset("test")

    X = np.load(X_tr).astype(np.float32)
    Y = np.load(Y_tr).astype(np.int32)

    label_map  = load_label_map()
    num_classes = len(label_map)
    log.info(f"Loaded {len(X)} samples  |  {num_classes} classes")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y
    )

    Y_train_cat = tf.keras.utils.to_categorical(Y_train, num_classes)
    Y_val_cat   = tf.keras.utils.to_categorical(Y_val,   num_classes)

    cw = compute_class_weight("balanced", classes=np.unique(Y_train), y=Y_train)
    class_weights = {i: w for i, w in enumerate(cw)}

    model = build_landmark_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(LANDMARK_H5_PATH), save_best_only=True, verbose=1),
    ]

    history = model.fit(
        X_train, Y_train_cat,
        validation_data=(X_val, Y_val_cat),
        epochs=EPOCHS_LM,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    _plot_history(history, "landmark")
    log.info(f"Landmark model saved → {LANDMARK_H5_PATH}")

    # evaluate on test set
    X_test = np.load(PROCESSED_DIR / "test_landmarks.npy").astype(np.float32)
    Y_test = np.load(PROCESSED_DIR / "test_labels.npy").astype(np.int32)
    Y_test_cat = tf.keras.utils.to_categorical(Y_test, num_classes)
    loss, acc = model.evaluate(X_test, Y_test_cat, verbose=0)
    log.info(f"Test accuracy: {acc:.4f}  |  Loss: {loss:.4f}")


# ── CNN Model ──────────────────────────────────────────────────────────────────
def build_cnn_model(num_classes: int) -> keras.Model:
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inp, out, name="cnn_model")
    return model


def train_cnn_model():
    log.info("═══ Training CNN Model ═══")
    from dataset_loader import image_generator, steps_per_epoch

    label_map   = load_label_map()
    num_classes = len(label_map)

    model = build_cnn_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=log.info)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1),
        keras.callbacks.ModelCheckpoint(str(MODEL_H5_PATH), save_best_only=True, verbose=1),
    ]

    spe_train = steps_per_epoch("train", BATCH_SIZE)
    spe_test  = steps_per_epoch("test",  BATCH_SIZE)

    history = model.fit(
        image_generator("train", label_map, BATCH_SIZE),
        steps_per_epoch=spe_train,
        validation_data=image_generator("test", label_map, BATCH_SIZE),
        validation_steps=spe_test,
        epochs=EPOCHS_CNN,
        callbacks=callbacks,
        verbose=1,
    )

    _plot_history(history, "cnn")
    log.info(f"CNN model saved → {MODEL_H5_PATH}")


# ── Plot ───────────────────────────────────────────────────────────────────────
def _plot_history(history, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="Train acc")
    axes[0].plot(history.history["val_accuracy"], label="Val acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"],     label="Train loss")
    axes[1].plot(history.history["val_loss"], label="Val loss")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    out = MODEL_DIR / f"{tag}_training_curves.png"
    plt.savefig(str(out))
    plt.close()
    log.info(f"Training curves saved → {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["landmark", "cnn", "both"],
        default="landmark",
        help="Which model to train",
    )
    args = parser.parse_args()

    if args.model in ("landmark", "both"):
        train_landmark_model()
    if args.model in ("cnn", "both"):
        train_cnn_model()