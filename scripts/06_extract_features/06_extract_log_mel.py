import sys
import os
import numpy as np
import librosa
import random
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_utils import extract_log_mel
from utils.augmentation_utils import apply_macbook_augment

DATASET_PATH = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
OUTPUT_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
AUG_SAMPLES_PATH = "/Volumes/Kingston XS1000 Media/project/augmented_samples"
SAMPLES_PER_CLASS = 3

SR = 16000


def process_log_mel():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(AUG_SAMPLES_PATH, exist_ok=True)

    saved_per_class = {}
    data = []
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

        print(f"Processing: {class_name} (Log-Mel)")
        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        random.shuffle(files)

        for idx, f in enumerate(files):
            if (idx + 1) % 500 == 0:
                print(f"  {idx + 1}/{len(files)}")

            file_path = os.path.join(class_dir, f)
            try:
                y, _ = librosa.load(file_path, sr=SR)
                if len(y) < SR:
                    continue
                segment = y[:SR]

                data.append(extract_log_mel(segment))
                labels.append(label_idx)

                if class_name != "noise":
                    aug_y = apply_macbook_augment(segment.copy(), noise_files, noise_path=noise_dir, sr=SR)

                    class_saved = saved_per_class.get(class_name, 0)
                    if class_saved < SAMPLES_PER_CLASS:
                        sf.write(os.path.join(AUG_SAMPLES_PATH, f"{class_name}_{class_saved:02d}_ORIGINAL.wav"), segment, SR)
                        sf.write(os.path.join(AUG_SAMPLES_PATH, f"{class_name}_{class_saved:02d}_AUGMENTED.wav"), aug_y, SR)
                        saved_per_class[class_name] = class_saved + 1

                    data.append(extract_log_mel(aug_y))
                    labels.append(label_idx)
            except Exception:
                continue

    if not data:
        print("Error: no samples processed.")
        return

    X = np.array(data)
    if len(X.shape) < 3:
        print(f"Error: unexpected shape {X.shape}")
        return

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    X = (X - X.min()) / (X.max() - X.min() + 1e-10)

    np.save(os.path.join(OUTPUT_PATH, "X_log_mel_full.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_log_mel_labels.npy"), np.array(labels))
    print(f"\nDone. Saved Log-Mel {X.shape} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    process_log_mel()
