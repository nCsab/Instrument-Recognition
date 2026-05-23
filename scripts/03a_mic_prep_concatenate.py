import os
import random
import soundfile as sf
import numpy as np

BASE_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/to_record"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000


def create_beep(duration_s, freq=440, sr=16000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


os.makedirs(OUTPUT_DIR, exist_ok=True)

for cat in CATEGORIES:
    cat_dir = os.path.join(BASE_DIR, cat, "train")
    if not os.path.exists(cat_dir):
        print(f"Warning: {cat_dir} not found.")
        continue

    files = [f for f in os.listdir(cat_dir) if f.endswith(".wav") and "clean" in f]
    files.sort()
    random.seed(42)
    random.shuffle(files)

    if not files:
        print(f"Warning: no clean files in {cat_dir}.")
        continue

    combined_audio = [create_beep(1.0, sr=SR), np.zeros(SR)]
    for f in files:
        data, _ = sf.read(os.path.join(cat_dir, f))
        combined_audio.append(data)
        combined_audio.append(np.zeros(int(SR * 0.2)))

    sf.write(os.path.join(OUTPUT_DIR, f"{cat}_for_mic.wav"), np.concatenate(combined_audio), SR)
    print(f"{cat}: {len(files)} samples concatenated")
