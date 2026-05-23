import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR_CLEAN = os.path.join(PROJECT_DIR, "hybrid_dataset_own_final")
DATASET_DIR_MIC = os.path.join(PROJECT_DIR, "hybrid_dataset_own_final_mic")


def print_dataset_stats(dataset_dir, title=""):
    if not os.path.exists(dataset_dir):
        return False

    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    print("\n" + "=" * 60)
    print(f"DATASET {title}")
    print(f"Path: {os.path.basename(dataset_dir)}")
    print("=" * 60)
    print(f"{'Class':<15} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("-" * 60)

    totals = [0, 0, 0]
    for cls in classes:
        counts = []
        for i, subset in enumerate(['train', 'val', 'test']):
            d = os.path.join(dataset_dir, cls, subset)
            n = len([f for f in os.listdir(d) if f.endswith('.wav')]) if os.path.exists(d) else 0
            counts.append(n)
            totals[i] += n
        print(f"{cls:<15} | {counts[0]:<7} | {counts[1]:<7} | {counts[2]:<7} | {sum(counts):<7}")

    print("-" * 60)
    print(f"{'TOTAL':<15} | {totals[0]:<7} | {totals[1]:<7} | {totals[2]:<7} | {sum(totals):<7}")
    print("=" * 60 + "\n")
    return True


def main():
    found = False
    if print_dataset_stats(DATASET_DIR_CLEAN, "CLEAN"):
        found = True
    if print_dataset_stats(DATASET_DIR_MIC, "MIC (AUGMENTED)"):
        found = True
    if not found:
        print(f"Error: no dataset found at:\n- {DATASET_DIR_CLEAN}\n- {DATASET_DIR_MIC}")


if __name__ == "__main__":
    main()
