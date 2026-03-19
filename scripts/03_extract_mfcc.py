import os
import numpy as np
import librosa
import random
import soundfile as sf
from utils.feature_utils import extract_mfcc, z_score_normalize
from utils.augmentation_utils import add_noise, add_reverb, add_eq, add_pitch_shift

# Útvonalak beállítása
DATASET_PATH = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset"
OUTPUT_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
CLASSES = ["guitar", "piano", "other", "noise"]
AUG_SAMPLES_PATH = "/Volumes/Kingston XS1000 Media/project/augmented_samples"
MAX_SAMPLES_TO_SAVE = 10 # Csak ennyit mentünk ki mutatóba egyenként

# Paraméterek
SR = 16000
DURATION = 1.0

def process_mfcc():
    """
    Kinyeri az MFCC jellemzőket az összes minta alapján a hybrid adatbázisból.
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(AUG_SAMPLES_PATH):
        os.makedirs(AUG_SAMPLES_PATH)

    # Számláló a kimentett mintákhoz
    saved_samples_count = 0

    data = []
    labels = []

    noise_dir = os.path.join(DATASET_PATH, "noise")
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"Processing class: {class_name} for MFCC...")
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
                
                # Alap MFCC + CMVN (Z-score) normalizálás
                mfcc_feat = extract_mfcc(segment)
                data.append(z_score_normalize(mfcc_feat))
                labels.append(label_idx)
                
                # Augmentálás (50% eséllyel)
                if class_name != "noise" and random.random() < 0.5:
                    aug_y = segment.copy()
                    choice = random.choice(["noise", "pitch", "reverb", "eq"])
                    if choice == "noise" and noise_files:
                        aug_y = add_noise(aug_y, random.choice(noise_files))
                    elif choice == "pitch":
                        aug_y = librosa.effects.pitch_shift(aug_y, sr=SR, n_steps=random.uniform(-1.5, 1.5))
                    elif choice == "reverb":
                        aug_y = add_reverb(aug_y, sr=SR)
                    elif choice == "eq":
                        aug_y = add_eq(aug_y)

                    # --- OPCIONÁLIS: Mentsük le a mintát, hogy belehallgathassunk ---
                    if saved_samples_count < MAX_SAMPLES_TO_SAVE:
                        aug_filename = f"aug_mfcc_{class_name}_{saved_samples_count:02d}_{choice}.wav"
                        sf.write(os.path.join(AUG_SAMPLES_PATH, aug_filename), aug_y, SR)
                        saved_samples_count += 1
                    
                    mfcc_feat_aug = extract_mfcc(aug_y)
                    data.append(z_score_normalize(mfcc_feat_aug))
                    labels.append(label_idx)
            except: continue

    X = np.array(data)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    # A Z-score már megtörtént mintánként, itt már nem kell globális min-max
    
    np.save(os.path.join(OUTPUT_PATH, "X_mfcc_full.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_mfcc_labels.npy"), np.array(labels))
    print(f"\nCOMPLETED! Saved MFCC as {X.shape} -> {OUTPUT_PATH}")

if __name__ == "__main__":
    process_mfcc()
