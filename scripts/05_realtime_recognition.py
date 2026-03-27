import os
import sys
import queue
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from collections import deque
from utils.feature_utils import extract_log_mel
from utils.model_utils import build_standard_2d_cnn_model

# --- KONFIGURÁCIÓ ---
MODEL_PATH = "/Volumes/Kingston XS1000 Media/project/models/best_log_mel_model.keras"
CLASSES = ["guitar", "piano", "vocal", "string", "noise"]
SR = 16000
WINDOW_DURATION = 1.0   # másodperc — ablak méret a feature kinyeréshez
STEP_DURATION = 0.5     # másodperc — frissítési ráta (overlap)
INPUT_SHAPE = (64, 32, 1)

# --- STABILIZÁLÁS ---
SMOOTHING_WINDOW = 4    # Ennyi utolsó predikciót átlagol (4 x 0.5s = 2 másodperc)
CONFIDENCE_THRESHOLD = 0.3  # Minimum magabiztosság az osztályváltáshoz
HYSTERESIS_BONUS = 0.05 # Bónusz az aktuálisan megjelenített osztálynak (ragadósság)

# --- SZÍNEK (ANSI escape kódok) ---
# guitar=beige/sárga, piano=fehér fekete háttérrel, vocal=polar blue, string=maroon, noise=szürke
CLASS_COLORS = {
    "guitar": "\033[93m",         # Világos sárga (beige-szerű)
    "piano":  "\033[97;40m",      # Fehér szöveg, fekete háttér
    "vocal":  "\033[96m",         # Világos ciánkék (polar blue)
    "string": "\033[38;5;88m",    # Maroon (sötét bordó)
    "noise":  "\033[90m",         # Szürke
}
RESET_COLOR = "\033[0m"

# Audio buffer queue
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback function for sounddevice InputStream."""
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())

def normalize_features(feat):
    """Normalize features matching the training normalization logic."""
    normalized = (feat - (-80.0)) / (0.0 - (-80.0) + 1e-10)
    return np.clip(normalized, 0.0, 1.0)

def display_prediction(smoothed_probs, current_class):
    """Print results with a color-coded bar chart in the same terminal line."""
    confidence = smoothed_probs[CLASSES.index(current_class)] * 100
    color = CLASS_COLORS.get(current_class, "")

    # Create simple bar
    bar_len = 20
    filled = int(bar_len * (confidence / 100.0))
    bar = "█" * filled + "░" * (bar_len - filled)

    # Main prediction with color
    output = f"\r{color}[ {bar} ] {current_class:7} ({confidence:5.1f}%){RESET_COLOR} | "
    # Add other classes with their own colors
    other_parts = []
    for i, cls in enumerate(CLASSES):
        if cls != current_class:
            c = CLASS_COLORS.get(cls, "")
            other_parts.append(f"{c}{cls[0].upper()}:{int(smoothed_probs[i]*100)}%{RESET_COLOR}")
    output += " ".join(other_parts)
    # Pad with spaces to clear previous longer lines
    output += "     "

    sys.stdout.write(output)
    sys.stdout.flush()

def main():
    print("--- VALÓS IDEJŰ HANGSZERFELISMERÉS (stabilizált) ---")
    print(f"  Simítás: {SMOOTHING_WINDOW} frame ({SMOOTHING_WINDOW * STEP_DURATION:.1f}s)")
    print(f"  Váltási küszöb: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"  Hiszterézis bónusz: +{HYSTERESIS_BONUS*100:.0f}%\n")

    # 1. Modell betöltése
    print(f"Modell betöltése... ({MODEL_PATH})")
    if not os.path.exists(MODEL_PATH):
        print(f"HIBA: A modell nem található: {MODEL_PATH}")
        return

    model = build_standard_2d_cnn_model(INPUT_SHAPE, len(CLASSES))
    try:
        model.load_weights(MODEL_PATH)
        print("Modell sikeresen betöltve.")
    except Exception as e:
        print(f"HIBA a súlyok betöltésekor: {e}")
        return

    # 2. Audio stream indítása
    print(f"Mikrofon indítása (SR={SR})...")
    stream = sd.InputStream(
        channels=1,
        samplerate=SR,
        callback=audio_callback,
        blocksize=int(SR * STEP_DURATION)
    )

    # Sliding window audio buffer
    full_buffer = np.zeros(int(SR * WINDOW_DURATION))

    # Stabilizáláshoz: utolsó N predikciós valószínűség-vektor
    prob_history = deque(maxlen=SMOOTHING_WINDOW)
    current_displayed_class = "noise"  # Induláskor noise

    with stream:
        print("\n🎧 Figyelés indítva... Nyomj Ctrl+C-t a leállításhoz.\n")
        try:
            while True:
                chunk = audio_q.get()
                chunk = chunk.flatten()

                # Sliding window frissítése
                full_buffer = np.roll(full_buffer, -len(chunk))
                full_buffer[-len(chunk):] = chunk

                # Feature kinyerés
                feat_raw = extract_log_mel(full_buffer, sr=SR)
                if feat_raw.shape[1] != INPUT_SHAPE[1]:
                    if feat_raw.shape[1] < INPUT_SHAPE[1]:
                        feat_raw = np.pad(feat_raw, ((0, 0), (0, INPUT_SHAPE[1] - feat_raw.shape[1])))
                    else:
                        feat_raw = feat_raw[:, :INPUT_SHAPE[1]]

                feat = normalize_features(feat_raw)
                X = feat.reshape(1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1)

                # Predikció
                pred_prob = model.predict(X, verbose=0)[0]
                prob_history.append(pred_prob)

                # --- STABILIZÁLÁS ---
                # 1. Mozgóátlag az utolsó N frame-ből
                smoothed = np.mean(prob_history, axis=0)

                # 2. Hiszterézis: az aktuálisan megjelenített osztály kap egy kis bónuszt
                #    hogy ne villogjon ide-oda két hasonló magabiztosságú osztály között
                adjusted = smoothed.copy()
                current_idx = CLASSES.index(current_displayed_class)
                adjusted[current_idx] += HYSTERESIS_BONUS

                # 3. Osztályváltás csak ha az új jelölt elég magabiztos
                new_idx = np.argmax(adjusted)
                new_class = CLASSES[new_idx]

                if new_class != current_displayed_class:
                    # Váltás csak ha a simított (nyers, bónusz nélküli) valószínűség
                    # meghaladja a küszöböt
                    if smoothed[new_idx] >= CONFIDENCE_THRESHOLD:
                        current_displayed_class = new_class

                # Megjelenítés
                display_prediction(smoothed, current_displayed_class)

        except KeyboardInterrupt:
            print("\n\n🛑 Leállítás...")
        except Exception as e:
            print(f"\nHIBA történt: {e}")

if __name__ == "__main__":
    main()
