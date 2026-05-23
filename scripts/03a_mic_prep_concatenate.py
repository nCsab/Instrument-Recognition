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


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for cat in CATEGORIES:
    cat_dir = os.path.join(BASE_DIR, cat, "train")
    if not os.path.exists(cat_dir):
        print(f"Warning: {cat_dir} not found. Please run 05_split_and_slice.py first to generate the train set.")
        continue
        
    files = [f for f in os.listdir(cat_dir) if f.endswith(".wav") and "clean" in f]
    files.sort()  # Először ABC sorrendbe, hogy a shuffle kiindulása mindig azonos legyen
    random.seed(42)
    random.shuffle(files)  # Kevert sorrend, de seed(42)-vel mindig ugyanaz a keverés
    
    if len(files) == 0:
        print(f"Warning: No clean files found in {cat_dir}.")
        continue
        
    # KIZÁRÓLAG a teljes train halmazt fűzzük össze a data leakage elkerülése és a maximális augmentáció érdekében
    selected_files = files

    combined_audio = [create_beep(1.0, sr=SR), np.zeros(SR)]

    for f in selected_files:
        data, _ = sf.read(os.path.join(cat_dir, f))
        combined_audio.append(data)
        combined_audio.append(np.zeros(int(SR * 0.2)))

    final_audio = np.concatenate(combined_audio)
    sf.write(os.path.join(OUTPUT_DIR, f"{cat}_for_mic.wav"), final_audio, SR)
    print(f"{cat} done ({len(selected_files)} samples, sync beep prepended)")
