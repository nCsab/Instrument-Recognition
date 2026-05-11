import os
import numpy as np
import librosa
import random
import soundfile as sf
from utils.feature_utils import extract_log_mel, extract_stft, extract_mfcc, z_score_normalize
from utils.augmentation_utils import apply_macbook_augment

DATASET_PATH = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
OUTPUT_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
AUG_SAMPLES_PATH = "/Volumes/Kingston XS1000 Media/project/augmented_samples"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
PREVIEW_PER_CLASS = 3

SR = 16000


def process_batch_dataset():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(AUG_SAMPLES_PATH, exist_ok=True)

    saved_per_class = {}
    data = {'log_mel': [], 'stft': [], 'mfcc': []}
    labels = []

    if not os.path.exists(DATASET_PATH):
        print(f"Error: dataset not found: {DATASET_PATH}")
        return

    noise_dir = os.path.join(DATASET_PATH, "noise")
    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        random.shuffle(files)
        print(f"Processing: {class_name} ({len(files)} files)")

        for idx, f in enumerate(files):
            if (idx + 1) % 500 == 0:
                print(f"  {idx + 1}/{len(files)}")

            file_path = os.path.join(class_dir, f)

            try:
                y, _ = librosa.load(file_path, sr=SR)
                if len(y) < SR:
                    continue

                segment = y[:SR]

                data['log_mel'].append(extract_log_mel(segment))
                data['stft'].append(extract_stft(segment))
                data['mfcc'].append(z_score_normalize(extract_mfcc(segment)))
                labels.append(label_idx)

                if class_name != "noise":
                    aug_y = apply_macbook_augment(segment.copy(), noise_files, noise_path=noise_dir, sr=SR)

                    class_saved = saved_per_class.get(class_name, 0)
                    if class_saved < PREVIEW_PER_CLASS:
                        sf.write(os.path.join(AUG_SAMPLES_PATH, f"{class_name}_{class_saved:02d}_ORIGINAL.wav"), segment, SR)
                        sf.write(os.path.join(AUG_SAMPLES_PATH, f"{class_name}_{class_saved:02d}_AUGMENTED.wav"), aug_y, SR)
                        saved_per_class[class_name] = class_saved + 1

                    data['log_mel'].append(extract_log_mel(aug_y))
                    data['stft'].append(extract_stft(aug_y))
                    data['mfcc'].append(z_score_normalize(extract_mfcc(aug_y)))
                    labels.append(label_idx)

            except Exception:
                continue

    if not labels:
        print("Error: no samples processed.")
        return

    print("\nSaving features...")
    for feat_name in ['log_mel', 'stft', 'mfcc']:
        X = np.array(data[feat_name])
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

        if feat_name != 'mfcc':
            X = (X - X.min()) / (X.max() - X.min() + 1e-10)

        save_path = os.path.join(OUTPUT_PATH, f"X_{feat_name}_full.npy")
        np.save(save_path, X)
        print(f"  {feat_name}: {X.shape} -> {save_path}")

    label_path = os.path.join(OUTPUT_PATH, "y_labels_full.npy")
    np.save(label_path, np.array(labels))
    print(f"  labels: {len(labels)} -> {label_path}")

    print("\nDone. Upload .npy files to Google Drive processed_data/ folder.")


if __name__ == "__main__":
    process_batch_dataset()
