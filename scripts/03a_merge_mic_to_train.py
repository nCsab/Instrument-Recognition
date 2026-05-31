import glob
import hashlib
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CLEAN_DIR = os.path.join(PROJECT_ROOT, "dataset_clean")
MIC_DIR = os.path.join(PROJECT_ROOT, "owndataset", "record")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiment_datasets")

SPLITS = ["train", "val", "test"]
INSTRUMENTS = ["guitar", "piano", "vocal", "string", "reed", "brass"]
CLASSES = INSTRUMENTS + ["noise"]

EXPERIMENTS = {
    "exp_clean": {
        "train": ["clean_train"], "val": ["clean_val"], "test": ["clean_test"],
    },
    "exp_augmented": {
        "train": ["clean_train"], "val": ["clean_val"], "test": ["clean_test"],
    },
    "exp_naive_deployment": {
        "train": ["clean_train"], "val": ["mic_val"], "test": ["mic_test"],
    },
    "exp_final": {
        "train": ["clean_train", "mic_train"], "val": ["mic_val"], "test": ["mic_test"],
    },
}


def wavs(path):
    return sorted(glob.glob(os.path.join(path, "*.wav")))


def file_hash(path):
    with open(path, "rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def source_files(cls, source):
    kind, split = source.split("_", 1)
    if kind == "clean":
        return wavs(os.path.join(CLEAN_DIR, cls, split))
    if cls == "noise":
        return []

    new_path = os.path.join(MIC_DIR, split, "slices", f"{cls}_mic_1sec")
    old_train_path = os.path.join(MIC_DIR, f"{cls}_mic_1sec")
    if os.path.exists(new_path):
        return wavs(new_path)
    if split == "train" and os.path.exists(old_train_path):
        return wavs(old_train_path)
    return []


def missing_mic_sources(config):
    missing = []
    for split in SPLITS:
        for source in config[split]:
            if not source.startswith("mic_"):
                continue
            classes = [cls for cls in INSTRUMENTS if not source_files(cls, source)]
            if classes:
                missing.append(f"{source}: {', '.join(classes)}")
    return missing


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def copy_class_split(exp_dir, cls, split, sources):
    dest = os.path.join(exp_dir, cls, split)
    os.makedirs(dest, exist_ok=True)

    copied = 0
    for source in sources:
        for index, file_path in enumerate(source_files(cls, source)):
            prefix = "mic" if source.startswith("mic_") else "clean"
            out_name = f"{cls}_{prefix}_{split}_{copied + index:04d}.wav"
            shutil.copy2(file_path, os.path.join(dest, out_name))
        copied = len(wavs(dest))
    return copied


def copy_balanced_noise(exp_dir, split, used_hashes):
    dest = os.path.join(exp_dir, "noise", split)
    os.makedirs(dest, exist_ok=True)

    counts = [len(wavs(os.path.join(exp_dir, cls, split))) for cls in INSTRUMENTS]
    target = int(sum(counts) / len(counts)) if counts else 0

    copied = 0
    current_hashes = set()
    for file_path in source_files("noise", f"clean_{split}"):
        hash_value = file_hash(file_path)
        if hash_value in used_hashes:
            continue
        shutil.copy2(file_path, os.path.join(dest, f"noise_clean_{split}_{copied:04d}.wav"))
        current_hashes.add(hash_value)
        copied += 1
        if copied == target:
            break

    used_hashes.update(current_hashes)
    return len(wavs(dest))


def build_experiment(name, config):
    exp_dir = os.path.join(EXPERIMENTS_DIR, name)
    missing = missing_mic_sources(config)

    print(f"\nBuilding {name}")
    if missing:
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        print("  Skipped because required microphone data is missing:")
        for item in missing:
            print(f"    - {item}")
        return

    reset_dir(exp_dir)
    for cls in INSTRUMENTS:
        for split in SPLITS:
            count = copy_class_split(exp_dir, cls, split, config[split])
            print(f"  {cls:<8} {split:<5} {count:4d} files")

    used_noise_hashes = set()
    for split in SPLITS:
        count = copy_balanced_noise(exp_dir, split, used_noise_hashes)
        print(f"  {'noise':<8} {split:<5} {count:4d} files")


def main():
    if not os.path.exists(CLEAN_DIR):
        raise FileNotFoundError(f"Missing clean dataset: {CLEAN_DIR}")
    for name, config in EXPERIMENTS.items():
        build_experiment(name, config)


if __name__ == "__main__":
    main()
