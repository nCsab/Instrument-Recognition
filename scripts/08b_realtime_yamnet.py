"""
08b_realtime_yamnet.py - Valós idejű hangszerfelismerés YAMNet Transfer Learning modellel

Ez a script a YAMNet-tel tanított modellt használja az élő felismeréshez.
A YAMNet közvetlenül a nyers hangból számítja ki a 1024-dimenziós embedding-et,
majd a mi kis osztályozó hálózatunk dönt a hangszer osztályról.

Használat: python3 scripts/08b_realtime_yamnet.py
"""
import os
import ssl

# Mac SSL hiba javítása (MINDEN MÁS IMPORT ELŐTT KELL LEGYEN!)
ssl._create_default_https_context = ssl._create_unverified_context
ssl.create_default_context = ssl._create_unverified_context

import sys
import queue
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
from collections import deque

MODEL_TYPE = 'micy'  # 'clean' vagy 'mic'
CLASSIFIER_PATH = f"/Volumes/Kingston XS1000 Media/project/models_{MODEL_TYPE}/best_yamnet_transfer_model.keras"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]

SR = 16000
WINDOW_DURATION = 1.0
STEP_DURATION = 0.25
EMBEDDING_DIM = 1024

SMOOTHING_WINDOW = 6
HYSTERESIS_BONUS = 0.05

THRESHOLDS = {
    "noise": 0.20,
    "default": 0.45
}

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


def build_classifier(embedding_dim=1024, num_classes=7):
    """A YAMNet embedding-ekre tanított osztályozó hálózat (meg kell egyeznie a tanítóval)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(embedding_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())


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
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"Classifier model not found: {CLASSIFIER_PATH}")
        print("Először tanítsd be a YAMNet modellt a Colab-ban (10_train_yamnet.py)!")
        return

    # YAMNet betöltése (ez néhány másodpercig tarthat)
    print("Loading YAMNet from TensorFlow Hub...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    print("YAMNet loaded!")

    # Saját osztályozó betöltése
    print("Loading classifier...")
    classifier = build_classifier(EMBEDDING_DIM, len(CLASSES))
    classifier.load_weights(CLASSIFIER_PATH)

    print(f"Smoothing: {SMOOTHING_WINDOW} frames ({SMOOTHING_WINDOW*STEP_DURATION:.1f}s)")
    print(f"Confidence thresholds - Noise: {THRESHOLDS['noise']*100:.0f}%, Instruments: {THRESHOLDS['default']*100:.0f}%")
    print(f"Hysteresis bonus: +{HYSTERESIS_BONUS*100:.0f}%\n")

    stream = sd.InputStream(
        channels=1,
        samplerate=SR,
        blocksize=int(SR * STEP_DURATION),
        callback=audio_callback
    )

    window_samples = int(SR * WINDOW_DURATION)
    full_buffer = np.zeros(window_samples, dtype=np.float32)
    prob_history = deque(maxlen=SMOOTHING_WINDOW)
    current_displayed_class = "noise"

    print("Listening... (Ctrl+C to stop)\n")

    try:
        with stream:
            while True:
                chunk = audio_q.get().flatten()

                full_buffer = np.roll(full_buffer, -len(chunk))
                full_buffer[-len(chunk):] = chunk

                # YAMNet embedding kinyerés a nyers hangból
                waveform_tf = tf.cast(full_buffer, tf.float32)
                scores, embeddings, spectrogram = yamnet_model(waveform_tf)

                # Átlagoljuk az embedding frame-eket
                mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
                X = mean_embedding.reshape(1, -1)

                # Osztályozó predikció
                pred_prob = classifier.predict(X, verbose=0)[0]
                prob_history.append(pred_prob)

                smoothed = np.mean(prob_history, axis=0)

                adjusted = smoothed.copy()
                current_idx = CLASSES.index(current_displayed_class)
                adjusted[current_idx] += HYSTERESIS_BONUS

                new_idx = np.argmax(adjusted)
                new_class = CLASSES[new_idx]

                if new_class != current_displayed_class:
                    required_thresh = THRESHOLDS["noise"] if new_class == "noise" else THRESHOLDS["default"]
                    if smoothed[new_idx] >= required_thresh:
                        current_displayed_class = new_class

                display_prediction(smoothed, current_displayed_class)

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
