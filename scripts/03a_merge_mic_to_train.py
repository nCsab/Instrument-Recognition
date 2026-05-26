import os, shutil, random, glob
import numpy as np

SOURCE_DIR = "/Users/csabanagy/Desktop/project/dataset_clean"
OUTPUT_DIR = "/Users/csabanagy/Desktop/project/dataset_mic"
MIC_DIR = "/Users/csabanagy/Desktop/project/owndataset/record"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]

def merge_class(cls):
    print(f"Merging: {cls}...")
    train_dir = os.path.join(OUTPUT_DIR, cls, "train")
    if not os.path.exists(train_dir): return

    group_nums = [0]
    for f in glob.glob(os.path.join(train_dir, "*_group*")):
        try:
            group_nums.append(int(os.path.basename(f).split("_group")[1].split("_")[0]))
        except (IndexError, ValueError):
            pass
    max_grp = max(group_nums)
    
    mic_clips = sorted(glob.glob(os.path.join(MIC_DIR, f"{cls}_mic_1sec", "*.wav")))
    for i, clip in enumerate(mic_clips):
        shutil.copy(clip, os.path.join(train_dir, f"{cls}_mic_group{max_grp + 1 + (i//5)}_clip{i%5:02d}.wav"))
    print(f"  {len(mic_clips)} mic clips -> train")

def balance_noise():
    print("\nBalancing noise class\n")
    
    clean_train = os.path.join(SOURCE_DIR, "noise", "train")
    backup_dir = os.path.join(SOURCE_DIR, "noise", "train_backup")
    
    # Create backup from clean_train if it doesn't exist yet
    if not os.path.exists(backup_dir) and os.path.exists(clean_train):
        shutil.copytree(clean_train, backup_dir)
        
    # Read the full pool of noise files from the backup directory
    pool = sorted(glob.glob(os.path.join(backup_dir, "*.wav")))
    
    avg_clean = int(np.mean([len(glob.glob(os.path.join(SOURCE_DIR, c, "train", "*.wav"))) for c in CLASSES]))
    avg_mic = int(np.mean([len(glob.glob(os.path.join(OUTPUT_DIR, c, "train", "*.wav"))) for c in CLASSES]))
    
    random.seed(42)
    random.shuffle(pool)
    
    for dest, count, lbl in [(clean_train, avg_clean * 2, "clean"), (os.path.join(OUTPUT_DIR, "noise", "train"), min(len(pool), avg_mic * 2), "mic")]:
        if os.path.exists(dest): shutil.rmtree(dest)
        os.makedirs(dest)
        for f in pool[:count]: shutil.copy2(f, os.path.join(dest, os.path.basename(f)))
        print(f"  {lbl}: {min(len(pool), count)} noise files")

def main():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)
    
    for cls in CLASSES: merge_class(cls)
    balance_noise()
    
    # Clean up the backup directory in OUTPUT_DIR so it doesn't remain in dataset_mic
    mic_backup = os.path.join(OUTPUT_DIR, "noise", "train_backup")
    if os.path.exists(mic_backup):
        shutil.rmtree(mic_backup)

if __name__ == "__main__":
    main()
