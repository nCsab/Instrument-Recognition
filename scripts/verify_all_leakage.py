import os
import re

DATASET_DIRS = [
    "/Volumes/Kingston XS1000 Media/project/dataset_clean",
    "/Volumes/Kingston XS1000 Media/project/dataset_mic"
]
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]

def get_group_id(filename):
    match = re.search(r"group(\d+)", filename)
    return int(match.group(1)) if match else None

def verify_dataset(dataset_dir):
    print("\n" + "=" * 60)
    print(f"DATA LEAKAGE CHECK: {os.path.basename(dataset_dir)}")
    print("=" * 60)

    total_leaks = 0

    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} does not exist.")
        return

    for cls in CLASSES:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        groups = {}
        for subset in ['train', 'val', 'test']:
            subset_dir = os.path.join(cls_dir, subset)
            groups[subset] = set()
            if os.path.exists(subset_dir):
                for f in os.listdir(subset_dir):
                    if f.endswith('.wav'):
                        gid = get_group_id(f)
                        if gid is not None:
                            groups[subset].add(gid)

        if not any(groups.values()):
            print(f"{cls:<10}: No files with group ID found.")
            continue

        leak_tv = groups['train'] & groups['val']
        leak_tt = groups['train'] & groups['test']
        leak_vt = groups['val'] & groups['test']
        class_leaks = len(leak_tv) + len(leak_tt) + len(leak_vt)
        total_leaks += class_leaks

        print(f"{cls:<10}: Train={len(groups['train'])}, Val={len(groups['val'])}, Test={len(groups['test'])}", end="")
        if class_leaks == 0:
            print(" -> OK")
        else:
            print(f" -> LEAK ({class_leaks} overlaps)")
            if leak_tv: print(f"    Train<->Val: {leak_tv}")
            if leak_tt: print(f"    Train<->Test: {leak_tt}")
            if leak_vt: print(f"    Val<->Test: {leak_vt}")

    print("=" * 60)
    print("NO DATA LEAKAGE DETECTED." if total_leaks == 0 else f"WARNING: {total_leaks} leak(s) found!")
    print("=" * 60)

if __name__ == "__main__":
    for d in DATASET_DIRS:
        verify_dataset(d)
