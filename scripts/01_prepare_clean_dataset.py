import os
import random
import re
import shutil

import librosa
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OWNDATASET_DIR = os.path.join(PROJECT_ROOT, "owndataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_clean")

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
SR = 16000
BLOCK_DURATION = 5.0
CLIP_DURATION = 1.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def wav_files(path):
    if not os.path.exists(path):
        return []
    return sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".wav"))


def source_prefix(cls, filename):
    if cls != "noise":
        parts = filename.split("_block")
        return parts[0] if len(parts) > 1 else cls
    if filename.startswith("esc50_"):
        return "esc50_noise"
    if filename.startswith("silence_"):
        return "silence"
    return "noise" if filename.startswith("noise_") else "other"


def group_blocks(cls, blocks):
    groups = {}
    for block in blocks:
        prefix = source_prefix(cls, os.path.basename(block))
        groups.setdefault(prefix, []).append(block)
    return groups


def split_blocks(groups):
    train, val, test = [], [], []
    random.seed(42)
    for prefix, blocks in groups.items():
        random.shuffle(blocks)
        n_train = int(len(blocks) * TRAIN_RATIO)
        n_val = int(len(blocks) * VAL_RATIO)
        train += blocks[:n_train]
        val += blocks[n_train:n_train + n_val]
        test += blocks[n_train + n_val:]
        print(f"  {prefix}: {len(blocks)} blocks -> Train={n_train}, Val={n_val}, Test={len(blocks) - n_train - n_val}")
    return {"train": train, "val": val, "test": test}


def write_clips(block_path, output_dir, prefix, group_id):
    audio, _ = librosa.load(block_path, sr=SR)
    clip_len = int(CLIP_DURATION * SR)
    max_clips = int(BLOCK_DURATION / CLIP_DURATION)
    for index in range(min(len(audio) // clip_len, max_clips)):
        start = index * clip_len
        clip = audio[start:start + clip_len]
        out_name = f"{prefix}_group{group_id}_clip{index:02d}.wav"
        sf.write(os.path.join(output_dir, out_name), clip, SR)


def build_class(cls):
    print(f"\nProcessing: {cls}...")
    blocks = wav_files(os.path.join(OWNDATASET_DIR, "instruments", cls))
    if not blocks:
        print(f"Error: no 5s blocks found for {cls}")
        return

    groups = group_blocks(cls, blocks)
    split_map = split_blocks(groups)
    print(f"Total blocks ({len(blocks)}): Train={len(split_map['train'])}, Val={len(split_map['val'])}, Test={len(split_map['test'])}")

    class_dir = os.path.join(OUTPUT_DIR, cls)
    if os.path.exists(class_dir):
        shutil.rmtree(class_dir)
    for split in split_map:
        os.makedirs(os.path.join(class_dir, split), exist_ok=True)

    group_id = 1
    for split, split_blocks_list in split_map.items():
        split_dir = os.path.join(class_dir, split)
        for block in split_blocks_list:
            write_clips(block, split_dir, f"{cls}_clean", group_id)
            group_id += 1


def group_id(filename):
    match = re.search(r"group\d+", filename)
    return match.group(0) if match else None


def files_by_group(*dirs):
    groups = {}
    for directory in dirs:
        for name in os.listdir(directory):
            if not name.endswith(".wav"):
                continue
            gid = group_id(name)
            if gid:
                groups.setdefault(gid, {"dir": directory, "files": []})["files"].append(name)
    return sorted(groups.values(), key=lambda item: len(item["files"]), reverse=True)


def rebalance_val_test(cls):
    val_dir = os.path.join(OUTPUT_DIR, cls, "val")
    test_dir = os.path.join(OUTPUT_DIR, cls, "test")
    if not os.path.exists(val_dir) or not os.path.exists(test_dir):
        return

    before = (len(wav_files(val_dir)), len(wav_files(test_dir)))
    targets = {val_dir: [], test_dir: []}
    counts = {val_dir: 0, test_dir: 0}

    for group in files_by_group(val_dir, test_dir):
        target_dir = val_dir if counts[val_dir] <= counts[test_dir] else test_dir
        targets[target_dir].append(group)
        counts[target_dir] += len(group["files"])

    moved = 0
    for target_dir, groups in targets.items():
        for group in groups:
            if group["dir"] == target_dir:
                continue
            for name in group["files"]:
                shutil.move(os.path.join(group["dir"], name), os.path.join(target_dir, name))
                moved += 1

    print(f"{cls:<10} | Before: Val={before[0]:<3} Test={before[1]:<3} | After: Val={counts[val_dir]:<3} Test={counts[test_dir]:<3} | Moved: {moved}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls in CLASSES:
        build_class(cls)

    print("\nBALANCING VAL/TEST SETS (BLOCK-LEVEL)\n")
    for cls in CLASSES:
        rebalance_val_test(cls)


if __name__ == "__main__":
    main()
