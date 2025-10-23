#!/usr/bin/env python3
"""
mnist_mlp.py - Train a small MLP on MNIST with improved structure, logging and CLI.

Improvements:
- argparse for configuration
- structured logging instead of casual prints
- proper tf.data pipeline (map -> cache -> shuffle -> batch -> prefetch)
- reproducibility via seeds
- callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
- save best model and final evaluation printout
- guarded main so file can be imported without side-effects
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import time
from typing import Tuple

import numpy as np
import tensorflow as tf

# Optional helper to reduce TF verbosity if the package is available
try:
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
except Exception:
    # silence_tensorflow is optional; ignore if not installed
    pass

import tensorflow_datasets as tfds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=24, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models and logs")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity for model.fit (0,1,2)")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    tf.random.set_seed(seed)
    np.random.seed(seed)


def get_datasets(batch_size: int, seed: int) -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    # Load MNIST
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        # Cast to float32 and scale to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # Ensure channel dimension for Keras models that expect channels
        if image.shape.rank == 2:
            image = tf.expand_dims(image, -1)
        return image, label

    AUTOTUNE = tf.data.AUTOTUNE

    # Training pipeline: map -> cache -> shuffle -> batch -> prefetch
    ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    # for full-shuffle use ds_info.splits['train'].num_examples, for memory constrained use a smaller buffer
    ds_train = ds_train.shuffle(buffer_size=ds_info.splits["train"].num_examples, seed=seed)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(AUTOTUNE)

    # Test pipeline: map -> batch -> cache -> prefetch
    ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(AUTOTUNE)

    return ds_train, ds_test, ds_info


def build_model(input_shape=(28, 28, 1)) -> tf.keras.Model:
    # A small MLP with dropout and modest capacity. Use from_logits=True on loss.
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),  # logits
        ]
    )
    return model


def configure_callbacks(model_dir: str) -> list:
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = os.path.join(model_dir, f"mnist_mlp_best_{timestamp}.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_sparse_categorical_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=6, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(model_dir, "logs", timestamp)),
    ]
    return callbacks


def main() -> None:
    args = parse_args()

    # configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    log = logging.getLogger("mnist_mlp")

    log.info("Starting training with args: %s", vars(args))
    set_seeds(args.seed)

    ds_train, ds_test, ds_info = get_datasets(args.batch_size, args.seed)

    model = build_model(input_shape=(28, 28, 1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
    )

    model.summary()

    callbacks = configure_callbacks(args.model_dir)

    t0 = time.perf_counter()
    history = model.fit(
        ds_train,
        epochs=args.epochs,
        validation_data=ds_test,
        validation_freq=1,
        callbacks=callbacks,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0
    log.info("Training completed in %.2f seconds", elapsed)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    log.info("Test loss: %.4f, Test accuracy: %.4f", test_loss, test_acc)

    # Save final model (even though checkpoint already saved best weights)
    final_path = os.path.join(args.model_dir, f"mnist_mlp_final_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    model.save(final_path)
    log.info("Saved final model to %s (best checkpoint saved separately)", final_path)

    # Optionally return history for programmatic use
    return


if __name__ == "__main__":
    main()