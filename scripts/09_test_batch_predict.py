import os
import numpy as np
import tensorflow as tf
import librosa
from utils.feature_utils import extract_log_mel

MODEL_PATH = "/Volumes/Kingston XS1000 Media/project/models/best_log_mel_2dcnn_model.keras"
TEST_DATA_DIR = "../model_test"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    model = build_model((64, 32, 1), len(CLASSES))
    model.load_weights(MODEL_PATH)

    files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith(".wav")]
    if not files:
        print("No test files found.")
        return

    for f in files:
        path = os.path.join(TEST_DATA_DIR, f)
        y, _ = librosa.load(path, sr=16000)
        
        expected = "unknown"
        for c in CLASSES:
            if c in f.lower():
                expected = c
                break

        print(f"\nFile: {f} (Expected: {expected})")
        
        for i in range(0, len(y) - 16000, 8000):
            segment = y[i:i+16000]
            feat = extract_log_mel(segment)
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-10)
            feat = feat.reshape(1, 64, 32, 1)
            
            pred = model.predict(feat, verbose=0)[0]
            idx = np.argmax(pred)
            print(f"  {i/16000:4.1f}s: {CLASSES[idx]:<10} ({pred[idx]:.2f})")


if __name__ == "__main__":
    main()
