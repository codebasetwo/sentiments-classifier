from typing import Any, List, Union

import numpy as np
import tensorflow as tf


class SentimentModel:
    def __init__(self, num_epochs: int = 3):
        self.num_epochs = num_epochs

    def create_model(
        self,
        loss: str = "sparse_categorical_crossentropy",
        optimizer: Union[str, Any] = "adam",
        metrics: List[str] = ["accuracy"],
        vocab_size: int = 5000,
        verbose: bool = False,
    ):

        # Model Definition with LSTM
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(vocab_size, 32),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

        # Set the training parameters
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        model.build(input_shape=(64, 32))

        if verbose:
            # Print the model summary
            model.summary()
        return model

    def fit_model(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
    ):

        history = model.fit(
            X_train,
            y_train,
            epochs=self.num_epochs,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
        )
        return history
