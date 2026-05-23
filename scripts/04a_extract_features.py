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


def process_batch_dataset(dataset_path, output_path, aug_samples_path, extract_raw=False):
    os.makedirs(output_path, exist_ok=True)
    if aug_samples_path:
        os.makedirs(aug_samples_path, exist_ok=True)

    saved_per_class = {}
    data = {
        'train': {'log_mel': [], 'stft': [], 'mfcc': [], 'raw': [], 'labels': []},
        'val':   {'log_mel': [], 'stft': [], 'mfcc': [], 'raw': [], 'labels': []},
        'test':  {'log_mel': [], 'stft': [], 'mfcc': [], 'raw': [], 'labels': []}
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
                    segment = segment[:SR]

                    data[subset]['log_mel'].append(extract_log_mel(segment))
                    data[subset]['stft'].append(extract_stft(segment))
                    data[subset]['mfcc'].append(z_score_normalize(extract_mfcc(segment)))
                    if extract_raw:
                        data[subset]['raw'].append(segment)
                    data[subset]['labels'].append(label_idx)

                    if subset == 'train' and class_name != 'noise' and "_mic_" not in f:
                        aug_y = apply_macbook_augment(segment.copy(), noise_files, noise_path=noise_dir, sr=SR)
                        if len(aug_y) < SR:
                            aug_y = np.pad(aug_y, (0, SR - len(aug_y)))
                        aug_y = aug_y[:SR]

                        if aug_samples_path:
                            class_saved = saved_per_class.get(class_name, 0)
                            if class_saved < PREVIEW_PER_CLASS:
                                sf.write(os.path.join(aug_samples_path, f"{class_name}_{class_saved:02d}_ORIGINAL.wav"), segment, SR)
                                sf.write(os.path.join(aug_samples_path, f"{class_name}_{class_saved:02d}_AUGMENTED.wav"), aug_y, SR)
                                saved_per_class[class_name] = class_saved + 1

                        data[subset]['log_mel'].append(extract_log_mel(aug_y))
                        data[subset]['stft'].append(extract_stft(aug_y))
                        data[subset]['mfcc'].append(z_score_normalize(extract_mfcc(aug_y)))
                        if extract_raw:
                            data[subset]['raw'].append(aug_y)
                        data[subset]['labels'].append(label_idx)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    print("\nSaving features...")
    for subset in ['train', 'val', 'test']:
        if not data[subset]['labels']:
            print(f"No data for {subset}.")
            continue

        print(f"\nSubset: {subset}")
        feats_to_save = ['log_mel', 'stft', 'mfcc']
        if extract_raw:
            feats_to_save.append('raw')

        for feat_name in feats_to_save:
            X = np.array(data[subset][feat_name])
            if feat_name != 'raw':
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

    clean_path = os.path.join(project_dir, "dataset_clean")
    clean_out = os.path.join(project_dir, "processed_data_clean")

    if os.path.exists(clean_path):
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION - CLEAN DATASET")
        print("=" * 60)
        process_batch_dataset(clean_path, clean_out, aug_samples_path=None, extract_raw=False)

    mic_path = os.path.join(project_dir, "dataset_mic")
    mic_out = os.path.join(project_dir, "processed_data_mic")

    if os.path.exists(mic_path):
        print("\n" + "=" * 60)
        print("FEATURE EXTRACTION - MIC DATASET (with raw segments for YAMNet)")
        print("=" * 60)
        aug_preview_path = os.path.join(project_dir, "augmented_preview")
        process_batch_dataset(mic_path, mic_out, aug_samples_path=aug_preview_path, extract_raw=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
