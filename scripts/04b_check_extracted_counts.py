import os

import numpy as np

# Ellenőrzi a feldolgozott NumPy tömböket.
# Kiírja a címkék osztályonkénti eloszlását és a Log-Mel feature alakját.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
SPLITS = ["train", "val", "test"]


def npy(path, default=None, mmap=None):
    return np.load(path, mmap_mode=mmap) if os.path.exists(path) else default


def npy_shape(path):
    if not os.path.exists(path):
        return "N/A"
    return np.load(path, mmap_mode="r").shape


def processed_dirs():
    if not os.path.exists(PROCESSED_DIR):
        return []
    return sorted(
        name for name in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, name))
    )


def print_stats(dataset_name):
    path = os.path.join(PROCESSED_DIR, dataset_name)
    labels = {split: npy(os.path.join(path, f"y_labels_{split}.npy"), np.array([])) for split in SPLITS}
    totals = [0, 0, 0]

    print(f"\n{dataset_name}\n{'CLASS':<15} | {'TRAIN':<7} | {'VAL':<7} | {'TEST':<7} | {'TOTAL':<7}")
    for index, cls in enumerate(CLASSES):
        counts = [int(np.sum(labels[split] == index)) if len(labels[split]) else 0 for split in SPLITS]
        totals = [a + b for a, b in zip(totals, counts)]
        print(f"{cls:<15} | {counts[0]:<7} | {counts[1]:<7} | {counts[2]:<7} | {sum(counts):<7}")

    print(f"{'TOTAL':<15} | {totals[0]:<7} | {totals[1]:<7} | {totals[2]:<7} | {sum(totals):<7}")
    print("Feature shapes (Log-Mel):")
    for split in SPLITS:
        shape = npy_shape(os.path.join(path, f"X_log_mel_{split}.npy"))
        print(f"  {split:<5}: {shape}")


def main():
    for dataset_name in processed_dirs():
        print_stats(dataset_name)


if __name__ == "__main__":
    main()
