import os
import sys
import queue
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from collections import deque
from utils.feature_utils import extract_log_mel

MODEL_TYPE = 'mic'  # 'clean' vagy 'mic'
MODEL_PATH = f"/Volumes/Kingston XS1000 Media/project/models_{MODEL_TYPE}/best_log_mel_2dcnn_model.keras"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]

SR = 16000
WINDOW_DURATION = 1.0
STEP_DURATION = 0.25  # 0.5-ről 0.25-re csökkentve: másodpercenként 4 predikció! (Ez működik "Időbeli TTA"-ként is)
INPUT_SHAPE = (128, 63, 1)

SMOOTHING_WINDOW = 6  # 4-ről 6-ra növelve, mivel 0.25s a lépés (így 1.5 másodpercet átlagol)
HYSTERESIS_BONUS = 0.05

# Adaptív küszöbértékek (noise-ra érzékenyebb, hangszerekre szigorúbb)
THRESHOLDS = {
    "noise": 0.20,      # A zajhoz elég 20% is
    "default": 0.45     # A hangszerekhez legalább 45% kell
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


# ============================================================
# SpecAugment réteg definíciója (szükséges a modell betöltéséhez)
# A súlyok betöltésekor a Keras-nak ismernie kell ezt az egyéni réteget.
# Inference (valós idejű felismerés) során ez a réteg automatikusan
# kikapcsol (training=False), tehát NEM módosítja a bemenetet.
# ============================================================
class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask_param=15, time_mask_param=8, num_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs

        augmented = inputs
        freq_dim = tf.shape(inputs)[1]
        time_dim = tf.shape(inputs)[2]

        for _ in range(self.num_masks):
            f = tf.random.uniform([], 1, self.freq_mask_param, dtype=tf.int32)
            f = tf.minimum(f, freq_dim)
            f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)
            indices = tf.range(freq_dim)
            freq_mask = tf.cast(tf.logical_or(indices < f0, indices >= f0 + f), tf.float32)
            freq_mask = tf.reshape(freq_mask, [1, -1, 1, 1])
            augmented = augmented * freq_mask

            t = tf.random.uniform([], 1, self.time_mask_param, dtype=tf.int32)
            t = tf.minimum(t, time_dim)
            t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)
            indices_t = tf.range(time_dim)
            time_mask = tf.cast(tf.logical_or(indices_t < t0, indices_t >= t0 + t), tf.float32)
            time_mask = tf.reshape(time_mask, [1, 1, -1, 1])
            augmented = augmented * time_mask

        return augmented

    def get_config(self):
        config = super().get_config()
        config.update({
            'freq_mask_param': self.freq_mask_param,
            'time_mask_param': self.time_mask_param,
            'num_masks': self.num_masks,
        })
        return config


def build_model(input_shape, num_classes):
    """Javított modell architektúra — meg kell egyeznie a tanító scriptben lévővel."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        SpecAugment(freq_mask_param=15, time_mask_param=8, num_masks=2),

        # 1. konvolúciós blokk
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # 2. konvolúciós blokk
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),

        # 3. konvolúciós blokk
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),

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

    print(f"Smoothing: {SMOOTHING_WINDOW} frames ({SMOOTHING_WINDOW*STEP_DURATION:.1f}s)")
    print(f"Confidence thresholds - Noise: {THRESHOLDS['noise']*100:.0f}%, Instruments: {THRESHOLDS['default']*100:.0f}%")
    print(f"Hysteresis bonus: +{HYSTERESIS_BONUS*100:.0f}%\n")

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
                    required_thresh = THRESHOLDS["noise"] if new_class == "noise" else THRESHOLDS["default"]
                    if smoothed[new_idx] >= required_thresh:
                        current_displayed_class = new_class

                display_prediction(smoothed, current_displayed_class)

        except KeyboardInterrupt:
            print("\n\nStopped.")


if __name__ == "__main__":
    main()
