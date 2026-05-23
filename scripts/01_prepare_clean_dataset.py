import os
import random
import shutil
import re
import librosa
import soundfile as sf
import numpy as np

OWNDATASET_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/dataset_clean"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000
BLOCK_DURATION = 5.0
CLIP_DURATION = 1.0

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def get_audio_files(directory):
    if not os.path.exists(directory): return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]


def get_group_id(filename):
    match = re.search(r"group\d+", filename)
    return match.group(0) if match else None


def slice_and_save(block_path, dest_dir, prefix, group_id):
    y, _ = librosa.load(block_path, sr=SR)
    clip_samples = int(CLIP_DURATION * SR)
    num_clips = min(len(y) // clip_samples, int(BLOCK_DURATION / CLIP_DURATION))

    saved_paths = []
    for i in range(num_clips):
        start = i * clip_samples
        clip = y[start:start + clip_samples]
        if np.max(np.abs(clip)) < 0.01:
            continue
        out_name = f"{prefix}_group{group_id}_clip{i:02d}.wav"
        out_path = os.path.join(dest_dir, out_name)
        sf.write(out_path, clip, SR)
        saved_paths.append(out_path)
    return saved_paths


def process_class(cls):
    print(f"\nProcessing: {cls}...")

    clean_blocks_dir = os.path.join(OWNDATASET_DIR, "instruments", cls, f"{cls}_5sec")
    clean_blocks = get_audio_files(clean_blocks_dir)
    clean_blocks.sort()

    if not clean_blocks:
        print(f"Error: no 5s blocks in {clean_blocks_dir}")
        return

    blocks_by_prefix = {}
    for block in clean_blocks:
        basename = os.path.basename(block)
        parts = basename.split("_block")
        prefix = parts[0] if len(parts) > 1 else cls
        blocks_by_prefix.setdefault(prefix, []).append(block)

    train_blocks, val_blocks, test_blocks = [], [], []

    random.seed(42)
    for prefix, blocks in blocks_by_prefix.items():
        random.shuffle(blocks)
        n = len(blocks)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_blocks.extend(blocks[:n_train])
        val_blocks.extend(blocks[n_train:n_train + n_val])
        test_blocks.extend(blocks[n_train + n_val:])

        print(f"  {prefix}: {n} blocks -> Train={n_train}, Val={n_val}, Test={n - n_train - n_val}")

    print(f"Total blocks ({len(clean_blocks)}): Train={len(train_blocks)}, Val={len(val_blocks)}, Test={len(test_blocks)}")

    cls_out_dir = os.path.join(OUTPUT_DIR, cls)
    if os.path.exists(cls_out_dir):
        shutil.rmtree(cls_out_dir)

    train_dir = os.path.join(cls_out_dir, "train")
    val_dir = os.path.join(cls_out_dir, "val")
    test_dir = os.path.join(cls_out_dir, "test")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    group_idx = 1
    for block in train_blocks:
        slice_and_save(block, train_dir, f"{cls}_clean", group_idx); group_idx += 1
    for block in val_blocks:
        slice_and_save(block, val_dir, f"{cls}_clean", group_idx); group_idx += 1
    for block in test_blocks:
        slice_and_save(block, test_dir, f"{cls}_clean", group_idx); group_idx += 1


def balance_class(cls):
    val_dir = os.path.join(OUTPUT_DIR, cls, "val")
    test_dir = os.path.join(OUTPUT_DIR, cls, "test")

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
    if os.path.exists(OUTPUT_DIR):
        for cls in CLASSES:
            d = os.path.join(OUTPUT_DIR, cls)
            if os.path.exists(d): shutil.rmtree(d)

    for cls in CLASSES:
        process_class(cls)

    print("\n" + "=" * 60)
    print("BALANCING VAL/TEST SETS (BLOCK-LEVEL)")
    print("=" * 60)
    for cls in CLASSES:
        balance_class(cls)
    print("=" * 60)
    print("Done.\n")


if __name__ == "__main__":
    main()
