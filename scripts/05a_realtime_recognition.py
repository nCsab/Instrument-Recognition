import glob
import os
import queue
import sys
from collections import deque

import numpy as np
import sounddevice as sd
import tensorflow as tf

from utils.feature_utils import extract_log_mel, normalize_db_feature

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_TYPE = "exp_final"  # Pl.: exp_clean, exp_augmented, exp_final
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", MODEL_TYPE)
CHECKPOINT_NAME = "exp_final_log_mel_2dcnn_val_20260527_222650_best_model.keras"  # None esetén a legfrissebb log_mel checkpointot választja.

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
CLASS_COLORS = {
    "guitar": "\033[93m",
    "piano": "\033[97;40m",
    "vocal": "\033[96m",
    "string": "\033[38;5;88m",
    "reed": "\033[38;5;208m",
    "brass": "\033[33m",
    "noise": "\033[90m",
}
RESET_COLOR = "\033[0m"

SR = 16000
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.25
INPUT_SHAPE = (128, 63, 1)
SMOOTHING_WINDOW = 6
HYSTERESIS_BONUS = 0.05
THRESHOLDS = {"noise": 0.20, "default": 0.45}

audio_q = queue.Queue()


class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask_param=15, time_mask_param=8, num_masks=2, apply_freq_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.apply_freq_mask = apply_freq_mask

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "num_masks": self.num_masks,
            "apply_freq_mask": self.apply_freq_mask,
        })
        return config


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())


def prepare_input(audio):
    feature = extract_log_mel(audio, sr=SR)
    target_frames = INPUT_SHAPE[1]
    if feature.shape[1] < target_frames:
        feature = np.pad(feature, ((0, 0), (0, target_frames - feature.shape[1])))
    else:
        feature = feature[:, :target_frames]
    return normalize_db_feature(feature).reshape(1, *INPUT_SHAPE)


def choose_class(probs, current_class):
    adjusted = probs.copy()
    adjusted[CLASSES.index(current_class)] += HYSTERESIS_BONUS
    candidate = CLASSES[int(np.argmax(adjusted))]
    if candidate == current_class:
        return current_class

    threshold = THRESHOLDS["noise"] if candidate == "noise" else THRESHOLDS["default"]
    return candidate if probs[CLASSES.index(candidate)] >= threshold else current_class


def display_prediction(probs, current_class):
    confidence = probs[CLASSES.index(current_class)] * 100
    filled = int(20 * confidence / 100)
    bar = "\u2588" * filled + "\u2591" * (20 - filled)
    color = CLASS_COLORS.get(current_class, "")

    others = []
    for index, cls in enumerate(CLASSES):
        if cls != current_class:
            others.append(f"{CLASS_COLORS.get(cls, '')}{cls[0].upper()}:{int(probs[index] * 100)}%{RESET_COLOR}")

    sys.stdout.write(f"\r{color}[ {bar} ] {current_class:7} ({confidence:5.1f}%){RESET_COLOR} | {' '.join(others)}     ")
    sys.stdout.flush()


def load_model():
    if CHECKPOINT_NAME:
        model_path = CHECKPOINT_NAME if os.path.isabs(CHECKPOINT_NAME) else os.path.join(MODEL_DIR, CHECKPOINT_NAME)
    else:
        checkpoints = sorted(glob.glob(os.path.join(MODEL_DIR, "*log_mel_2dcnn_*_best_model.keras")))
        if not checkpoints:
            checkpoints = sorted(glob.glob(os.path.join(MODEL_DIR, "best_log_mel_2dcnn_model.keras")))
        model_path = checkpoints[-1] if checkpoints else None

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found in: {MODEL_DIR}")
    print(f"Using checkpoint: {model_path}")
    return tf.keras.models.load_model(model_path, custom_objects={"SpecAugment": SpecAugment}, compile=False)


def main():
    try:
        print("Loading model...")
        model = load_model()
    except Exception as error:
        print(error)
        return

    print(f"Smoothing: {SMOOTHING_WINDOW} frames ({SMOOTHING_WINDOW * STEP_SECONDS:.1f}s)")
    print(f"Thresholds - Noise: {THRESHOLDS['noise'] * 100:.0f}%, Instruments: {THRESHOLDS['default'] * 100:.0f}%")
    print(f"Hysteresis: +{HYSTERESIS_BONUS * 100:.0f}%\n")

    buffer = np.zeros(int(SR * WINDOW_SECONDS), dtype=np.float32)
    history = deque(maxlen=SMOOTHING_WINDOW)
    current_class = "noise"

    stream = sd.InputStream(
        channels=1,
        samplerate=SR,
        blocksize=int(SR * STEP_SECONDS),
        callback=audio_callback,
    )

    print("Listening... (Ctrl+C to stop)\n")
    try:
        with stream:
            while True:
                chunk = audio_q.get().flatten()
                buffer = np.roll(buffer, -len(chunk))
                buffer[-len(chunk):] = chunk

                probs = model.predict(prepare_input(buffer), verbose=0)[0]
                history.append(probs)
                smoothed = np.mean(history, axis=0)
                current_class = choose_class(smoothed, current_class)
                display_prediction(smoothed, current_class)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()
