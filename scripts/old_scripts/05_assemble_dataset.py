import os
import shutil
import random

OWNDATASET_ROOT = "/Volumes/Kingston XS1000 Media/project/owndataset"
RECORDED_DIR = os.path.join(OWNDATASET_ROOT, "recorded_from_mic")
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
SAMPLES_PER_CLASS = 1500


def assemble():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    for cls in CLASSES:
        if cls == "noise":
            continue
        
        # 1. Copy clean samples
        clean_dir = os.path.join(OWNDATASET_ROOT, cls, f"{cls}_1sec")
        if os.path.exists(clean_dir):
            clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
            print(f"Copying {len(clean_files)} clean samples for {cls}...")
            for f in clean_files:
                shutil.copy(os.path.join(clean_dir, f), os.path.join(OUTPUT_DIR, cls, f))
        else:
            print(f"Warning: Clean dir {clean_dir} not found.")

        # 2. Copy mic samples
        mic_dir = os.path.join(RECORDED_DIR, f"{cls}_mic_1sec")
        if os.path.exists(mic_dir):
            mic_files = [f for f in os.listdir(mic_dir) if f.endswith(".wav")]
            random.shuffle(mic_files)
            to_copy = min(len(mic_files), 250)
            print(f"Copying {to_copy} mic samples for {cls}...")
            for f in mic_files[:to_copy]:
                shutil.copy(os.path.join(mic_dir, f), os.path.join(OUTPUT_DIR, cls, f))
        else:
            print(f"Warning: Mic dir {mic_dir} not found.")

    noise_src = "/Volumes/Kingston XS1000 Media/project/dataset/ESC-50-master/audio"
    if os.path.exists(noise_src):
        noise_files = [f for f in os.listdir(noise_src) if f.endswith(".wav")]
        random.shuffle(noise_files)
        to_copy = min(len(noise_files), SAMPLES_PER_CLASS)
        print(f"Copying {to_copy} samples for noise...")
        for i, f in enumerate(noise_files[:to_copy]):
            shutil.copy(os.path.join(noise_src, f), os.path.join(OUTPUT_DIR, "noise", f"noise_{i:04d}.wav"))

    print("\nDataset assembled.")


if __name__ == "__main__":
    assemble()
