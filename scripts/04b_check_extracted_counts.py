import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR_CLEAN = os.path.join(PROJECT_DIR, "processed_data_clean")
PROCESSED_DIR_MIC = os.path.join(PROJECT_DIR, "processed_data_mic")

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]


def print_feature_stats(processed_dir, title=""):
    if not os.path.exists(processed_dir):
        return False

    print("\n" + "=" * 70)
    print(f"EXTRACTED FEATURES {title}")
    print(f"Path: {os.path.basename(processed_dir)}")
    print("=" * 70)
    print(f"{'Class':<15} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 70)

    labels = {}
    for subset in ['train', 'val', 'test']:
        path = os.path.join(processed_dir, f"y_labels_{subset}.npy")
        labels[subset] = np.load(path) if os.path.exists(path) else np.array([])

    totals = [0, 0, 0]
    for idx, cls in enumerate(CLASSES):
        counts = []
        for i, subset in enumerate(['train', 'val', 'test']):
            n = int(np.sum(labels[subset] == idx)) if len(labels[subset]) > 0 else 0
            counts.append(n)
            totals[i] += n
        print(f"{cls:<15} | {counts[0]:<7} | {counts[1]:<7} | {counts[2]:<7} | {sum(counts):<7}")

    print("-" * 70)
    print(f"{'TOTAL':<15} | {totals[0]:<7} | {totals[1]:<7} | {totals[2]:<7} | {sum(totals):<7}")

    print("-" * 70)
    print("Feature shapes (Log-Mel):")
    for subset in ['train', 'val', 'test']:
        path = os.path.join(processed_dir, f"X_log_mel_{subset}.npy")
        if os.path.exists(path):
            shape = np.load(path, mmap_mode='r').shape
            print(f"  {subset:<5}: {shape}")
        else:
            print(f"  {subset:<5}: N/A")

    print("=" * 70 + "\n")
    return True


def main():
    found = False
    if print_feature_stats(PROCESSED_DIR_CLEAN, "CLEAN"):
        found = True
    if print_feature_stats(PROCESSED_DIR_MIC, "MIC"):
        found = True
    if not found:
        print(f"Error: no data found at:\n- {PROCESSED_DIR_CLEAN}\n- {PROCESSED_DIR_MIC}")


if __name__ == "__main__":
    main()
