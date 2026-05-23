import os
import numpy as np
import librosa
import random
import soundfile as sf
from utils.feature_utils import extract_log_mel, extract_stft, extract_mfcc, z_score_normalize
from utils.augmentation_utils import apply_macbook_augment

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
PREVIEW_PER_CLASS = 3
SR = 16000

def process_batch_dataset(dataset_path, output_path, aug_samples_path):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(aug_samples_path, exist_ok=True)

    saved_per_class = {}
    
    # Adatstruktúra a train, val, test halmazoknak
    data = {
        'train': {'log_mel': [], 'stft': [], 'mfcc': [], 'labels': []},
        'val':   {'log_mel': [], 'stft': [], 'mfcc': [], 'labels': []},
        'test':  {'log_mel': [], 'stft': [], 'mfcc': [], 'labels': []}
    }

    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found: {dataset_path}")
        return

    # Noise fájlok összegyűjtése az augmentációhoz (csak a train halmazból vesszük, hogy tiszta legyen a szeparáció)
    noise_dir = os.path.join(dataset_path, "noise", "train")
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]

    for label_idx, class_name in enumerate(CLASSES):
        class_root = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_root):
            print(f"Warning: {class_root} not found, skipping.")
            continue

        for subset in ['train', 'val', 'test']:
            subset_dir = os.path.join(class_root, subset)
            if not os.path.exists(subset_dir):
                continue
                
            files = [f for f in os.listdir(subset_dir) if f.endswith(".wav")]
            random.shuffle(files)
            
            # Különleges kezelés a zaj train adatok kiegyenlítésére (augmentáció nélkül)
            is_noise_train = (class_name == "noise" and subset == 'train')
            
            if is_noise_train:
                # Kiszámoljuk a hangszeres train adatok átlagát a feature tömbben
                instrument_labels = [l for l in data['train']['labels'] if l < 6]
                if instrument_labels:
                    target_noise_train = len(instrument_labels) // 6
                else:
                    target_noise_train = len(files) # fallback
                
                print(f"Noise train kiegyenlítés (augmentáció nélkül): Cél = {target_noise_train} (Elérhető a lemezen: {len(files)})")
                
                if target_noise_train <= len(files):
                    files_to_process = files[:target_noise_train]
                else:
                    # Ha több kell mint ami van, nyers duplikációval pótoljuk (nem augmentációval)
                    extra_needed = target_noise_train - len(files)
                    extra_files = random.choices(files, k=extra_needed)
                    files_to_process = files + extra_files
            else:
                files_to_process = files
            
            print(f"Processing: {class_name} - {subset} ({len(files_to_process)} mintát generálunk)")

            for idx, f in enumerate(files_to_process):
                if (idx + 1) % 500 == 0 or (idx + 1) == len(files_to_process):
                    print(f"  {idx + 1}/{len(files_to_process)}")

                file_path = os.path.join(subset_dir, f)

                try:
                    y, _ = librosa.load(file_path, sr=SR)
                    if len(y) < SR:
                        continue

                    segment = y[:SR]

                    # Nyers klip hozzáadása az adott halmazhoz
                    data[subset]['log_mel'].append(extract_log_mel(segment))
                    data[subset]['stft'].append(extract_stft(segment))
                    data[subset]['mfcc'].append(z_score_normalize(extract_mfcc(segment)))
                    data[subset]['labels'].append(label_idx)

                    # Augmentálás train halmaz esetén hangszer osztályokhoz
                    # Augmentálás train halmaz esetén hangszer osztályokhoz
                    # Fontos: A valódi mikrofonos felvételeket ("_mic_") NEM augmentáljuk újra, hogy elkerüljük a dupla torzítást!
                    if subset == 'train' and class_name != 'noise' and "_mic_" not in f:
                        aug_y = apply_macbook_augment(segment.copy(), noise_files, noise_path=noise_dir, sr=SR)

                        # Hangszerosztályoknál elmentjük a preview-t
                        class_saved = saved_per_class.get(class_name, 0)
                        if class_saved < PREVIEW_PER_CLASS:
                            sf.write(os.path.join(aug_samples_path, f"{class_name}_{class_saved:02d}_ORIGINAL.wav"), segment, SR)
                            sf.write(os.path.join(aug_samples_path, f"{class_name}_{class_saved:02d}_AUGMENTED.wav"), aug_y, SR)
                            saved_per_class[class_name] = class_saved + 1

                        data[subset]['log_mel'].append(extract_log_mel(aug_y))
                        data[subset]['stft'].append(extract_stft(aug_y))
                        data[subset]['mfcc'].append(z_score_normalize(extract_mfcc(aug_y)))
                        data[subset]['labels'].append(label_idx)

                except Exception as e:
                    print(f"Hiba {file_path} feldolgozása közben: {e}")
                    continue

    print("\nSaving features...")
    for subset in ['train', 'val', 'test']:
        if not data[subset]['labels']:
            print(f"Nincsenek feldolgozott adatok a(z) {subset} halmazban.")
            continue
            
        print(f"\nHalmaz: {subset}")
        for feat_name in ['log_mel', 'stft', 'mfcc']:
            X = np.array(data[subset][feat_name])
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

            if feat_name != 'mfcc':
                X = (X - X.min()) / (X.max() - X.min() + 1e-10)

            save_path = os.path.join(output_path, f"X_{feat_name}_{subset}.npy")
            np.save(save_path, X)
            print(f"  {feat_name}: {X.shape} -> {save_path}")

        label_path = os.path.join(output_path, f"y_labels_{subset}.npy")
        np.save(label_path, np.array(data[subset]['labels']))
        print(f"  labels: {len(data[subset]['labels'])} -> {label_path}")

def main():
    project_dir = "/Volumes/Kingston XS1000 Media/project"
    
    # 1. Feldolgozás a tiszta adathalmazra
    clean_path = os.path.join(project_dir, "hybrid_dataset_own_final")
    clean_out = os.path.join(project_dir, "processed_data_clean")
    clean_aug = os.path.join(project_dir, "augmented_samples_clean")
    
    if os.path.exists(clean_path):
        print("\n" + "="*70)
        print("FEATURE EXTRACTION - TISZTA ADATHALMAZ (CSAK INTERNETES)")
        print("="*70)
        process_batch_dataset(clean_path, clean_out, clean_aug)
        print(f"\nTiszta adathalmaz feldolgozása kész! Mentve: {clean_out}")

    # 2. Feldolgozás a mikrofonos adathalmazra
    mic_path = os.path.join(project_dir, "hybrid_dataset_own_final_mic")
    mic_out = os.path.join(project_dir, "processed_data_mic")
    mic_aug = os.path.join(project_dir, "augmented_samples_mic")
    
    if os.path.exists(mic_path):
        print("\n" + "="*70)
        print("FEATURE EXTRACTION - MIKROFONOS AUGMENTÁLT ADATHALMAZ")
        print("="*70)
        process_batch_dataset(mic_path, mic_out, mic_aug)
        print(f"\nMikrofonos adathalmaz feldolgozása kész! Mentve: {mic_out}")

    print("\n" + "="*70)
    print("Minden feldolgozás sikeresen befejeződött!")
    print("Most már feltöltheted a 'processed_data_clean' és 'processed_data_mic' mappák tartalmát a Google Drive-ra a Colab tanításhoz.")
    print("="*70)

if __name__ == "__main__":
    main()
