import os
import re
import shutil

DATASET_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]


def get_group_id(filename):
    match = re.search(r"group\d+", filename)
    return match.group(0) if match else None


def balance_class(cls):
    val_dir = os.path.join(DATASET_DIR, cls, "val")
    test_dir = os.path.join(DATASET_DIR, cls, "test")

    if not os.path.exists(val_dir) or not os.path.exists(test_dir):
        print(f"Error: {cls} - val or test directory not found.")
        return

    val_files = [f for f in os.listdir(val_dir) if f.endswith('.wav')]
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
    initial_val, initial_test = len(val_files), len(test_files)

    blocks = {}
    for f in val_files:
        gid = get_group_id(f)
        if gid:
            blocks.setdefault(gid, {"current_dir": val_dir, "files": []})["files"].append(f)
    for f in test_files:
        gid = get_group_id(f)
        if gid:
            blocks.setdefault(gid, {"current_dir": test_dir, "files": []})["files"].append(f)

    sorted_blocks = sorted(blocks.items(), key=lambda x: len(x[1]["files"]), reverse=True)

    new_val, new_test = [], []
    val_count, test_count = 0, 0

    for gid, info in sorted_blocks:
        n = len(info["files"])
        if val_count <= test_count:
            new_val.append((gid, info)); val_count += n
        else:
            new_test.append((gid, info)); test_count += n

    moved = 0
    for _, info in new_val:
        if info["current_dir"] == test_dir:
            for f in info["files"]:
                shutil.move(os.path.join(test_dir, f), os.path.join(val_dir, f)); moved += 1
    for _, info in new_test:
        if info["current_dir"] == val_dir:
            for f in info["files"]:
                shutil.move(os.path.join(val_dir, f), os.path.join(test_dir, f)); moved += 1

    print(f"{cls:<10} | Before: Val={initial_val:<3} Test={initial_test:<3} | After: Val={val_count:<3} Test={test_count:<3} | Moved: {moved}")


def main():
    print("\n" + "=" * 60)
    print("BALANCING VAL/TEST SETS (BLOCK-LEVEL)")
    print("=" * 60)
    for cls in CLASSES:
        balance_class(cls)
    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
