import hashlib
import os
import random
import shutil

import librosa
import numpy as np
import soundfile as sf

from utils.augmentation_utils import apply_macbook_augment
from utils.feature_utils import extract_log_mel, extract_mfcc, extract_stft, normalize_db_feature, z_score_normalize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiment_datasets")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")
AUGMENTED_PREVIEW_DIR = os.path.join(PROJECT_ROOT, "augmented_previews")

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
SPLITS = ["train", "val", "test"]
FEATURES = {
    "log_mel": extract_log_mel,
    "stft": extract_stft,
    "mfcc": lambda y: z_score_normalize(extract_mfcc(y)),
}
EXPERIMENTS = [
    ("exp_clean", ["log_mel", "stft", "mfcc"], False, None, False, "1. CLEAN BASELINE"),
    ("exp_augmented", ["log_mel"], False, "exp_augmented", True, "2. AUGMENTED BASELINE"),
    ("exp_naive_deployment", ["log_mel"], False, "exp_naive_deployment", True, "3. NAIVE DEPLOYMENT"),
    ("exp_final", ["log_mel"], True, "exp_final", True, "4. FINAL SYSTEM"),
]
SR = 16000
PREVIEW_PER_CLASS = 3

random.seed(42)
np.random.seed(42)


def wavs(path):
    if not os.path.exists(path):
        return []
    return sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".wav"))


def one_second(audio):
    return np.pad(audio, (0, max(0, SR - len(audio))))[:SR]


def audio_hash(audio):
    return hashlib.md5(np.ascontiguousarray(audio).tobytes()).hexdigest()


def add_sample(data, split, label, audio, feature_names, extract_raw):
    for name in feature_names:
        extractor = FEATURES[name]
        data[split][name].append(extractor(audio))
    if extract_raw:
        data[split]["raw"].append(audio)
    data[split]["labels"].append(label)


def save_arrays(data, output_dir, feature_names, extract_raw):
    output_names = feature_names + (["raw"] if extract_raw else [])
    print("\nSaving features...")

    for split in SPLITS:
        if not data[split]["labels"]:
            continue
        print(f"\nSubset: {split}")

        for name in output_names:
            array = np.array(data[split][name])
            if name != "raw":
                array = array.reshape(array.shape[0], array.shape[1], array.shape[2], 1)
                if name != "mfcc":
                    array = normalize_db_feature(array)

            path = os.path.join(output_dir, f"X_{name}_{split}.npy")
            np.save(path, array)
            print(f"  {name}: {array.shape} -> {path}")

        labels = np.array(data[split]["labels"])
        path = os.path.join(output_dir, f"y_labels_{split}.npy")
        np.save(path, labels)
        print(f"  labels: {len(labels)} -> {path}")


def process_dataset(dataset_dir, output_dir, feature_names, preview_dir=None, do_augmentation=True, extract_raw=True):
    if not os.path.exists(dataset_dir):
        print(f"Error: {dataset_dir} not found")
        return

    unknown = sorted(set(feature_names) - set(FEATURES))
    if unknown:
        raise ValueError(f"Unknown feature types: {unknown}")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if preview_dir:
        if os.path.exists(preview_dir):
            shutil.rmtree(preview_dir)
        os.makedirs(preview_dir, exist_ok=True)

    names = feature_names + (["raw"] if extract_raw else []) + ["labels"]
    data = {split: {name: [] for name in names} for split in SPLITS}
    seen_noise = {split: set() for split in SPLITS}
    preview_count = {}
    noise_dir = os.path.join(dataset_dir, "noise", "train")
    noise_files = wavs(noise_dir)

    for label, cls in enumerate(CLASSES):
        for split in SPLITS:
            files = wavs(os.path.join(dataset_dir, cls, split))
            random.shuffle(files)

            if cls == "noise":
                target = sum(item < 6 for item in data[split]["labels"]) // 6
                if target:
                    files = files[:target] if target <= len(files) else files + random.choices(files, k=target - len(files))

            print(f"Processing: {cls} - {split} ({len(files)} samples)")
            for file_path in files:
                try:
                    audio, _ = librosa.load(file_path, sr=SR)
                    if len(audio) < SR:
                        continue
                    audio = one_second(audio)

                    if cls == "noise":
                        earlier = set().union(*(seen_noise[s] for s in SPLITS[:SPLITS.index(split)]))
                        hash_value = audio_hash(audio)
                        if hash_value in earlier:
                            continue
                        seen_noise[split].add(hash_value)

                    add_sample(data, split, label, audio, feature_names, extract_raw)

                    can_augment = do_augmentation and split == "train" and cls != "noise" and "_mic_" not in os.path.basename(file_path)
                    if can_augment:
                        aug = apply_macbook_augment(audio.copy(), noise_files, noise_path=noise_dir, sr=SR)
                        aug = one_second(aug)
                        add_sample(data, split, label, aug, feature_names, extract_raw)

                        count = preview_count.get(cls, 0)
                        if preview_dir and count < PREVIEW_PER_CLASS:
                            sf.write(os.path.join(preview_dir, f"{cls}_{count:02d}_ORIGINAL.wav"), audio, SR)
                            sf.write(os.path.join(preview_dir, f"{cls}_{count:02d}_AUGMENTED.wav"), aug, SR)
                            preview_count[cls] = count + 1
                except Exception as error:
                    print(f"Error {os.path.basename(file_path)}: {error}")

    save_arrays(data, output_dir, feature_names, extract_raw)


def main():
    for dataset_name, feature_names, extract_raw, preview_name, do_aug, title in EXPERIMENTS:
        dataset_dir = os.path.join(EXPERIMENTS_DIR, dataset_name)
        if not os.path.exists(dataset_dir):
            continue

        saved_outputs = feature_names + (["raw"] if extract_raw else [])
        print(f"\nFEATURE EXTRACTION - {title} ({', '.join(saved_outputs)})\n")
        process_dataset(
            dataset_dir,
            os.path.join(PROCESSED_DIR, dataset_name),
            feature_names,
            os.path.join(AUGMENTED_PREVIEW_DIR, preview_name) if preview_name else None,
            do_aug,
            extract_raw=extract_raw,
        )


if __name__ == "__main__":
    main()
