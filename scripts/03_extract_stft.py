import os
import numpy as np
import librosa
import random
import soundfile as sf
from utils.feature_utils import extract_stft
from utils.augmentation_utils import add_noise, add_reverb, add_eq, add_pitch_shift, apply_macbook_augment

# Útvonalak beállítása
DATASET_PATH = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
OUTPUT_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
CLASSES = ["guitar", "piano", "vocal", "other", "noise"]
AUG_SAMPLES_PATH = "/Volumes/Kingston XS1000 Media/project/augmented_samples"
MAX_SAMPLES_TO_SAVE = 10 # Csak ennyit mentünk ki mutatóba egyenként

# Paraméterek
SR = 16000
DURATION = 1.0

def process_stft():
    """
    Kinyeri az STFT spektrogramokat az összes minta alapján a hybrid adatbázisból.
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(AUG_SAMPLES_PATH):
        os.makedirs(AUG_SAMPLES_PATH)

    # Számláló a kimentett mintákhoz
    saved_samples_count = 0

    data = []
    labels = []

    if not os.path.exists(DATASET_PATH):
        print(f"HIBA: A dataset mappa nem található: {DATASET_PATH}")
        return

    noise_dir = os.path.join(DATASET_PATH, "noise")
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"Processing class: {class_name} for STFT...")
        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        random.shuffle(files)
        
        for idx, f in enumerate(files):
            if (idx + 1) % 500 == 0:
                print(f"    Processed {idx + 1}/{len(files)} files...")
                
            file_path = os.path.join(class_dir, f)
            try:
                y, _ = librosa.load(file_path, sr=SR)
                if len(y) < SR: continue
                segment = y[:SR]
                
                # Alap STFT
                data.append(extract_stft(segment))
                labels.append(label_idx)
                
                # MacBook Augmentáció (100% eséllyel, kivéve a zaj kategóriát)
                if class_name != "noise":
                    aug_y = segment.copy()
                    aug_y = apply_macbook_augment(aug_y, noise_files, noise_path=noise_dir, sr=SR)

                    # --- OPCIONÁLIS: Mentsük le a mintát, hogy belehallgathassunk ---
                    if saved_samples_count < MAX_SAMPLES_TO_SAVE:
                        aug_filename = f"aug_stft_{class_name}_{saved_samples_count:02d}_macbook.wav"
                        sf.write(os.path.join(AUG_SAMPLES_PATH, aug_filename), aug_y, SR)
                        saved_samples_count += 1
                    
                    data.append(extract_stft(aug_y))
                    labels.append(label_idx)
            except: continue

    if not data:
        print("HIBA: Nem sikerült egyetlen mintát sem feldolgozni! Ellenőrizd az elérési utakat.")
        return

    X = np.array(data)
    if len(X.shape) < 3:
        print(f"HIBA: Rossz adat-dimenzió: {X.shape}. Valószínűleg üres a dataset.")
        return

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = (X - X.min()) / (X.max() - X.min() + 1e-10)
    
    np.save(os.path.join(OUTPUT_PATH, "X_stft_full.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_stft_labels.npy"), np.array(labels))
    print(f"\nCOMPLETED! Saved STFT as {X.shape} -> {OUTPUT_PATH}")

if __name__ == "__main__":
    process_stft()
