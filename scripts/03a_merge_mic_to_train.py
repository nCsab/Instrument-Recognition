import os
import shutil
import random

OWNDATASET_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset"
SOURCE_DIR = "/Volumes/Kingston XS1000 Media/project/dataset_clean"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/dataset_mic"
MIC_DIR = os.path.join(OWNDATASET_DIR, "record", "recorded_from_mic")
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]


def get_audio_files(directory):
    if not os.path.exists(directory): return []
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')])


def merge_class(cls):
    print(f"\nMerging: {cls}...")
    train_dir = os.path.join(OUTPUT_DIR, cls, "train")
    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found.")
        return

    train_files = get_audio_files(train_dir)
    max_group = 0
    for f in train_files:
        basename = os.path.basename(f)
        if "_group" in basename:
            try:
                group_num = int(basename.split("_group")[1].split("_")[0])
                max_group = max(max_group, group_num)
            except: pass

    group_idx = max_group + 1

    mic_clips_dir = os.path.join(MIC_DIR, f"{cls}_mic_1sec")
    mic_clips = get_audio_files(mic_clips_dir)

    if mic_clips:
        print(f"  {len(mic_clips)} mic clips -> train")
        for i, clip in enumerate(mic_clips):
            mic_group = group_idx + (i // 5)
            out_name = f"{cls}_mic_group{mic_group}_clip{i%5:02d}.wav"
            shutil.copy(clip, os.path.join(train_dir, out_name))
    else:
        print(f"  Warning: no mic clips in {mic_clips_dir}")


def balance_noise():
    print("\n--- Noise class balancing ---")
    project_dir = "/Volumes/Kingston XS1000 Media/project"
    noise_pool = os.path.join(project_dir, "noise_train_pool")
    clean_noise_train = os.path.join(SOURCE_DIR, "noise", "train")
    mic_noise_train = os.path.join(OUTPUT_DIR, "noise", "train")

    os.makedirs(noise_pool, exist_ok=True)

    if os.path.exists(clean_noise_train):
        clean_files = [f for f in os.listdir(clean_noise_train) if f.endswith(".wav")]
        pool_files = [f for f in os.listdir(noise_pool) if f.endswith(".wav")]
        if len(pool_files) < len(clean_files):
            print(f"Copying {len(clean_files)} noise files to pool...")
            for f in clean_files:
                dst = os.path.join(noise_pool, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(clean_noise_train, f), dst)

    pool_files = sorted([f for f in os.listdir(noise_pool) if f.endswith(".wav")])
    if not pool_files:
        print("Error: noise pool is empty!")
        return

    # Target counts based on instrument class averages
    clean_counts = []
    for cls in CLASSES:
        d = os.path.join(SOURCE_DIR, cls, "train")
        if os.path.exists(d):
            clean_counts.append(len([f for f in os.listdir(d) if f.endswith(".wav")]))
    n_clean = int(round(sum(clean_counts) / len(clean_counts))) if clean_counts else 346

    mic_counts = []
    for cls in CLASSES:
        d = os.path.join(OUTPUT_DIR, cls, "train")
        if os.path.exists(d):
            mic_counts.append(len([f for f in os.listdir(d) if f.endswith(".wav")]))
    n_mic = int(round(sum(mic_counts) / len(mic_counts))) if mic_counts else 690

    n_clean_noise = n_clean * 2
    n_mic_noise = min(len(pool_files), n_mic * 2)

    print(f"Targets - Clean: {n_clean_noise}, Mic: {n_mic_noise}")

    random.seed(42)
    shuffled = list(pool_files)
    random.shuffle(shuffled)

    selected_mic = shuffled[:n_mic_noise]
    selected_clean = shuffled[:n_clean_noise]

    for target_dir, selected, label in [(clean_noise_train, selected_clean, "clean"), (mic_noise_train, selected_mic, "mic")]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        for f in selected:
            shutil.copy2(os.path.join(noise_pool, f), os.path.join(target_dir, f))
        print(f"  {label}: {len(selected)} noise files")

    print("Noise balancing done.")


def main():
    print("--- Merging mic recordings into train set ---")

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: {SOURCE_DIR} not found.")
        return

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print(f"Copying {SOURCE_DIR} -> {OUTPUT_DIR}...")
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)

    for cls in CLASSES:
        merge_class(cls)

    balance_noise()
    print("\nDone.")


if __name__ == "__main__":
    main()
