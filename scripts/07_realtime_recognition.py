import os
import sys
import queue
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from collections import deque
from utils.feature_utils import extract_log_mel

MODEL_PATH = "/Volumes/Kingston XS1000 Media/project/models/best_log_mel_2dcnn_model.keras"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]

SR = 16000
WINDOW_DURATION = 1.0
STEP_DURATION = 0.5
INPUT_SHAPE = (64, 32, 1)

SMOOTHING_WINDOW = 4
CONFIDENCE_THRESHOLD = 0.3
HYSTERESIS_BONUS = 0.05

CLASS_COLORS = {
    "guitar": "\033[93m",
    "piano":  "\033[97;40m",
    "vocal":  "\033[96m",
    "string": "\033[38;5;88m",
    "reed":   "\033[38;5;208m",
    "brass":  "\033[33m",
    "noise":  "\033[90m",
}
RESET_COLOR = "\033[0m"

audio_q = queue.Queue()


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


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())


def normalize_features(feat):
    normalized = (feat - (-80.0)) / (0.0 - (-80.0) + 1e-10)
    return np.clip(normalized, 0.0, 1.0)


def display_prediction(smoothed_probs, current_class):
    confidence = smoothed_probs[CLASSES.index(current_class)] * 100
    color = CLASS_COLORS.get(current_class, "")

    bar_len = 20
    filled = int(bar_len * (confidence / 100.0))
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    output = f"\r{color}[ {bar} ] {current_class:7} ({confidence:5.1f}%){RESET_COLOR} | "

    other_parts = []
    for i, cls in enumerate(CLASSES):
        if cls != current_class:
            c = CLASS_COLORS.get(cls, "")
            other_parts.append(f"{c}{cls[0].upper()}:{int(smoothed_probs[i]*100)}%{RESET_COLOR}")
    output += " ".join(other_parts) + "     "

    sys.stdout.write(output)
    sys.stdout.flush()


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    print("Loading model...")
    model = build_model(INPUT_SHAPE, len(CLASSES))
    model.load_weights(MODEL_PATH)

    print(f"Smoothing: {SMOOTHING_WINDOW} frames ({SMOOTHING_WINDOW * STEP_DURATION:.1f}s)")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Hysteresis bonus: +{HYSTERESIS_BONUS*100:.0f}%")

    stream = sd.InputStream(
        channels=1,
        samplerate=SR,
        callback=audio_callback,
        blocksize=int(SR * STEP_DURATION)
    )

    full_buffer = np.zeros(int(SR * WINDOW_DURATION))
    prob_history = deque(maxlen=SMOOTHING_WINDOW)
    current_displayed_class = "noise"

    with stream:
        print("\nListening... (Ctrl+C to stop)\n")
        try:
            while True:
                chunk = audio_q.get().flatten()

                full_buffer = np.roll(full_buffer, -len(chunk))
                full_buffer[-len(chunk):] = chunk

                feat_raw = extract_log_mel(full_buffer, sr=SR)
                if feat_raw.shape[1] < INPUT_SHAPE[1]:
                    feat_raw = np.pad(feat_raw, ((0, 0), (0, INPUT_SHAPE[1] - feat_raw.shape[1])))
                elif feat_raw.shape[1] > INPUT_SHAPE[1]:
                    feat_raw = feat_raw[:, :INPUT_SHAPE[1]]

                feat = normalize_features(feat_raw)
                X = feat.reshape(1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)

                pred_prob = model.predict(X, verbose=0)[0]
                prob_history.append(pred_prob)

                smoothed = np.mean(prob_history, axis=0)

                adjusted = smoothed.copy()
                current_idx = CLASSES.index(current_displayed_class)
                adjusted[current_idx] += HYSTERESIS_BONUS

                new_idx = np.argmax(adjusted)
                new_class = CLASSES[new_idx]

                if new_class != current_displayed_class:
                    if smoothed[new_idx] >= CONFIDENCE_THRESHOLD:
                        current_displayed_class = new_class

                display_prediction(smoothed, current_displayed_class)

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    main()
