import os
import numpy as np
import librosa
import random
import soundfile as sf
from utils.feature_utils import extract_log_mel, extract_stft, extract_mfcc, z_score_normalize
from utils.augmentation_utils import add_noise, add_reverb, add_eq, add_pitch_shift, apply_macbook_augment

# Útvonalak beállítása
DATASET_PATH = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
OUTPUT_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
CLASSES = ["guitar", "piano", "vocal", "string", "noise"]
AUG_SAMPLES_PATH = "/Volumes/Kingston XS1000 Media/project/augmented_samples"
MAX_SAMPLES_TO_SAVE = 20 # Csak ennyit mentünk ki mutatóba

# Paraméterek (Fixen a régi 03_extract-tel megegyezően)
SR = 16000
DURATION = 1.0

def process_batch_dataset():
    """
    Kinyeri mindhárom feature típust (Log-Mel, STFT, MFCC) az összes minta alapján,
    hogy utána a Colab-ban villámgyors legyen a tanítás.
    """
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    if not os.path.exists(AUG_SAMPLES_PATH):
        os.makedirs(AUG_SAMPLES_PATH)

    # Számláló a kimentett mintákhoz
    saved_samples_count = 0

    # Adatgyűjtők
    data = {'log_mel': [], 'stft': [], 'mfcc': []}
    labels = []

    if not os.path.exists(DATASET_PATH):
        print(f"HIBA: A dataset mappa nem található: {DATASET_PATH}")
        return

    # Zajfájlok betöltése az augmentációhoz
    noise_dir = os.path.join(DATASET_PATH, "noise")
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping.")
            continue
            
        print(f"Processing class: {class_name}")
        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        random.shuffle(files)
        
        # A 02_prepare 5000/2000 mintát gyűjtött ki. Most mindet feldolgozzuk.
        print(f"  Found {len(files)} files. Extracting features...")
        
        for idx, f in enumerate(files):
            if (idx + 1) % 500 == 0:
                print(f"    Processed {idx + 1}/{len(files)} files...")
                
            file_path = os.path.join(class_dir, f)
            
            try:
                # Beolvasás
                y, _ = librosa.load(file_path, sr=SR)
                
                # Szeletelés (csak az első 1 másodperc, hogy fix hosszat kapjunk a modellnek)
                if len(y) < SR:
                    continue # Túl rövid fájl
                
                segment = y[:SR]
                
                # 1. Alap features (Log-Mel, STFT, MFCC)
                data['log_mel'].append(extract_log_mel(segment))
                data['stft'].append(extract_stft(segment))
                
                # MFCC-Specifikus: Azonnali Z-normalizálás
                mfcc_feat = extract_mfcc(segment)
                data['mfcc'].append(z_score_normalize(mfcc_feat))
                
                labels.append(label_idx)
                
                # 2. MacBook Augmentáció (100% eséllyel, kivéve a zaj kategóriát)
                # Minden tiszta mintához generálunk egy "élő" mikrofonos szimulációt is,
                # így megkétszerezzük a hasznos adathalmazt.
                if class_name != "noise":
                    aug_y = segment.copy()
                    aug_y = apply_macbook_augment(aug_y, noise_files, noise_path=noise_dir, sr=SR)

                    # --- OPCIONÁLIS: Mentsük le a mintát, hogy belehallgathassunk ---
                    if saved_samples_count < MAX_SAMPLES_TO_SAVE:
                        aug_filename = f"aug_{class_name}_{saved_samples_count:02d}_macbook.wav"
                        sf.write(os.path.join(AUG_SAMPLES_PATH, aug_filename), aug_y, SR)
                        saved_samples_count += 1

                    # Kimentjük az augmentált változatot is mindhárom módon
                    data['log_mel'].append(extract_log_mel(aug_y))
                    data['stft'].append(extract_stft(aug_y))
                    
                    mfcc_feat_aug = extract_mfcc(aug_y)
                    data['mfcc'].append(z_score_normalize(mfcc_feat_aug))
                    
                    labels.append(label_idx)
                    
            except Exception as e:
                # print(f"Error processing {f}: {e}")
                continue

    if not labels:
        print("HIBA: Nem sikerült egyetlen mintát sem feldolgozni! Ellenőrizd az elérési utakat.")
        return

    # Mentés .npy fájlokba (külön-külön a hiba elkerülése végett)
    print("\nSaving feature sets to disk...")
    for feat_name in ['log_mel', 'stft', 'mfcc']:
        X = np.array(data[feat_name])
        # Alakítsuk át 4D-sre a 2D CNN-hez
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Log-Mel és STFT esetén maradhat a globális min-max, 
        # az MFCC esetében már megtörtént a Z-score mintánként.
        if feat_name != 'mfcc':
            X = (X - X.min()) / (X.max() - X.min() + 1e-10)
        
        save_path = os.path.join(OUTPUT_PATH, f"X_{feat_name}_full.npy")
        np.save(save_path, X)
        print(f"  Saved {feat_name} as {X.shape} -> {save_path}")
    
    label_path = os.path.join(OUTPUT_PATH, "y_labels_full.npy")
    np.save(label_path, np.array(labels))
    print(f"  Saved labels as {len(labels)} -> {label_path}")

    print("\nCOMPLETED! Now you can upload these 4 .npy files to Google Drive processed_data/ folder.")

if __name__ == "__main__":
    process_batch_dataset()
