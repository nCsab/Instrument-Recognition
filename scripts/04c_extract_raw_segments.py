import os
import numpy as np
import librosa
import random
from utils.augmentation_utils import apply_macbook_augment

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
SR = 16000


def process_batch_dataset(dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    data = {
        'train': {'raw': [], 'labels': []},
        'val':   {'raw': [], 'labels': []},
        'test':  {'raw': [], 'labels': []}
    }

    if not os.path.exists(dataset_path):
        print(f"Error: dataset not found: {dataset_path}")
        return

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

            is_noise_train = (class_name == "noise" and subset == 'train')

            if is_noise_train:
                instrument_labels = [l for l in data['train']['labels'] if l < 6]
                target = len(instrument_labels) // 6 if instrument_labels else len(files)
                print(f"Noise train balancing: target={target}, available={len(files)}")

                if target <= len(files):
                    files_to_process = files[:target]
                else:
                    files_to_process = files + random.choices(files, k=target - len(files))
            else:
                files_to_process = files

            print(f"Processing: {class_name} - {subset} ({len(files_to_process)} samples)")

            for idx, f in enumerate(files_to_process):
                if (idx + 1) % 500 == 0 or (idx + 1) == len(files_to_process):
                    print(f"  {idx + 1}/{len(files_to_process)}")

                file_path = os.path.join(subset_dir, f)
                try:
                    y, _ = librosa.load(file_path, sr=SR)
                    if len(y) < SR:
                        continue

                    segment = y[:SR]
                    if len(segment) < SR:
                        segment = np.pad(segment, (0, SR - len(segment)))

                    data[subset]['raw'].append(segment)
                    data[subset]['labels'].append(label_idx)

                    if subset == 'train' and class_name != 'noise' and "_mic_" not in f:
                        aug_y = apply_macbook_augment(segment.copy(), noise_files, noise_path=noise_dir, sr=SR)
                        if len(aug_y) < SR:
                            aug_y = np.pad(aug_y, (0, SR - len(aug_y)))
                        aug_y = aug_y[:SR]
                        data[subset]['raw'].append(aug_y)
                        data[subset]['labels'].append(label_idx)

                except Exception as e:
                    print(f"Error {file_path}: {e}")
                    continue

    print("\nSaving raw audio segments...")
    for subset in ['train', 'val', 'test']:
        if not data[subset]['labels']:
            print(f"No data for {subset}.")
            continue

        X = np.array(data[subset]['raw'], dtype=np.float32)
        y = np.array(data[subset]['labels'])

        x_path = os.path.join(output_path, f"X_raw_{subset}.npy")
        np.save(x_path, X)
        np.save(os.path.join(output_path, f"y_labels_{subset}.npy"), y)
        print(f"  {subset}: X={X.shape}, y={y.shape} -> {x_path}")


def main():
    project_dir = "/Volumes/Kingston XS1000 Media/project"
    mic_path = os.path.join(project_dir, "hybrid_dataset_own_final_mic")
    mic_out = os.path.join(project_dir, "processed_data_mic")

    if os.path.exists(mic_path):
        print("\n" + "=" * 60)
        print("RAW AUDIO EXTRACTION - MIC DATASET (for YAMNet)")
        print("=" * 60)
        process_batch_dataset(mic_path, mic_out)
    else:
        print(f"Error: {mic_path} not found!")

    print("\nDone.")


if __name__ == "__main__":
    main()
