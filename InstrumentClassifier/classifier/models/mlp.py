import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras


class MLPStrategy:

    def __init__(self, epochs=80, batch_size=32, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.num_classes = None

    def _build_model(self, input_dim, num_classes):
        tf.random.set_seed(self.random_state)
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(self, X_train, y_train):
        if X_train.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_train.shape}.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        self.num_classes = len(np.unique(y_train))
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = self._build_model(X_scaled.shape[1], self.num_classes)

        self.model.fit(
            X_scaled, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True,
                ),
            ],
            verbose=0,
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled, verbose=0)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0)

    def get_name(self):
        return "MLP"
