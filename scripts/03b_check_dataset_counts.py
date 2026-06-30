import os

# Egyszerű ellenőrző script a WAV-darabszámokhoz.
# Nem módosít adatot, csak kiírja a datasetek osztályonkénti train/val/test
# eloszlását.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiment_datasets")
SPLITS = ["train", "val", "test"]


def count_wavs(path):
    if not os.path.exists(path):
        return 0
    return sum(file_name.endswith(".wav") for file_name in os.listdir(path))


def datasets():
    items = [("dataset_clean", os.path.join(PROJECT_ROOT, "dataset_clean"))]
    if os.path.exists(EXPERIMENTS_DIR):
        items += [
            (f"experiment_datasets/{name}", os.path.join(EXPERIMENTS_DIR, name))
            for name in sorted(os.listdir(EXPERIMENTS_DIR))
            if os.path.isdir(os.path.join(EXPERIMENTS_DIR, name))
        ]
    return items


def print_stats(title, dataset_dir):
    if not os.path.exists(dataset_dir):
        return

    classes = sorted(
        item for item in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, item))
    )

    totals = [0, 0, 0]
    print(f"\n{title}\n{'CLASS':<15} | {'TRAIN':<7} | {'VAL':<7} | {'TEST':<7} | {'TOTAL':<7}")
    for cls in classes:
        counts = [count_wavs(os.path.join(dataset_dir, cls, split)) for split in SPLITS]
        totals = [old + new for old, new in zip(totals, counts)]
        print(f"{cls:<15} | {counts[0]:<7} | {counts[1]:<7} | {counts[2]:<7} | {sum(counts):<7}")
    print(f"{'TOTAL':<15} | {totals[0]:<7} | {totals[1]:<7} | {totals[2]:<7} | {sum(totals):<7}")


def main():
    for title, path in datasets():
        print_stats(title, path)


if __name__ == "__main__":
    main()
