
import os
import shutil
import random
import csv
from collections import defaultdict

ROOT_DIR = "/Volumes/Kingston XS1000 Media/project/dataset"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset"

# Target classes
CLASSES = ["guitar", "piano", "other", "noise"]
SAMPLES_PER_CLASS = 5000

AUDIO_EXTENSIONS = ('.wav', '.flac')

def get_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, f))
    return audio_files

def collect_guitar_sources():
    sources = []
    # IRMAS
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData/gel")))
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData/gac")))
    
    # Medley-solos-DB
    medley_dir = os.path.join(ROOT_DIR, "Medley-solos-DB")
    csv_file = os.path.join(medley_dir, "Medley-solos-DB_metadata.csv")
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'guitar' in row.get('instrument', '').lower():
                    # Pattern for medley filenames: Medley-solos-DB_subset-instID_uuid.wav
                    uuid = row.get('uuid4')
                    subset = row.get('subset')
                    inst_id = row.get('instrument_id')
                    filename = f"Medley-solos-DB_{subset}-{inst_id}_{uuid}.wav"
                    path = os.path.join(medley_dir, filename)
                    if os.path.exists(path):
                        sources.append(path)

    # NSynth
    nsynth_audio = os.path.join(ROOT_DIR, "nsynth-train/audio")
    if os.path.exists(nsynth_audio):
        for f in os.listdir(nsynth_audio):
            if f.startswith("guitar_") and f.endswith(".wav"):
                sources.append(os.path.join(nsynth_audio, f))
    
    return sources

def collect_piano_sources():
    sources = []
    # IRMAS
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData/pia")))
    
    # Medley-solos-DB
    medley_dir = os.path.join(ROOT_DIR, "Medley-solos-DB")
    csv_file = os.path.join(medley_dir, "Medley-solos-DB_metadata.csv")
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'piano' in row.get('instrument', '').lower():
                    uuid = row.get('uuid4')
                    subset = row.get('subset')
                    inst_id = row.get('instrument_id')
                    filename = f"Medley-solos-DB_{subset}-{inst_id}_{uuid}.wav"
                    path = os.path.join(medley_dir, filename)
                    if os.path.exists(path):
                        sources.append(path)

    # NSynth
    nsynth_audio = os.path.join(ROOT_DIR, "nsynth-train/audio")
    if os.path.exists(nsynth_audio):
        for f in os.listdir(nsynth_audio):
            if (f.startswith("keyboard_") or f.startswith("piano_")) and f.endswith(".wav"):
                sources.append(os.path.join(nsynth_audio, f))
                
    return sources

def collect_other_sources():
    sources = []
    # IRMAS (non-guitar, non-piano)
    other_irmas = ["cel", "cla", "flu", "org", "sax", "tru", "vio", "voi"]
    for d in other_irmas:
        sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData", d)))
        
    # TinySOL
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "TinySOL")))
    
    # good-sounds (excluding those we might use as targets if duplicated, but usually they are separate)
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "good-sounds")))
    
    # NSynth (others)
    nsynth_audio = os.path.join(ROOT_DIR, "nsynth-train/audio")
    if os.path.exists(nsynth_audio):
        # We'll actually iterate os.listdir once to be faster if we needed, but for clarity:
        for f in os.listdir(nsynth_audio):
            if f.endswith(".wav") and not (f.startswith("guitar_") or f.startswith("keyboard_") or f.startswith("piano_")):
                if random.random() < 0.1: # 10% chance to reduce list size
                    sources.append(os.path.join(nsynth_audio, f))

    return sources

def collect_noise_sources():
    # ESC-50
    sources = get_audio_files(os.path.join(ROOT_DIR, "ESC-50-master/audio"))
    return sources

def prepare_hybrid():
    print("Starting Hybrid Dataset Preparation...")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    # Collection mapping
    collection_funcs = {
        "guitar": collect_guitar_sources,
        "piano": collect_piano_sources,
        "other": collect_other_sources,
        "noise": collect_noise_sources
    }

    for cls, func in collection_funcs.items():
        print(f"Collecting sources for: {cls}...")
        all_sources = func()
        print(f"  Found {len(all_sources)} potential samples.")
        
        sample_size = min(len(all_sources), SAMPLES_PER_CLASS)
        selected = random.sample(all_sources, sample_size)
        
        print(f"  Copying {sample_size} samples...")
        for i, src in enumerate(selected):
            ext = os.path.splitext(src)[1]
            dest = os.path.join(OUTPUT_DIR, cls, f"{cls}_{i:04d}{ext}")
            shutil.copy(src, dest)
            if (i+1) % 500 == 0:
                print(f"    Copied {i+1} samples...")

    print("\nSUCCESS! Hybrid dataset created at:")
    print(OUTPUT_DIR)
    
    # Simple verification
    print("\nVerification:")
    for cls in CLASSES:
        count = len(os.listdir(os.path.join(OUTPUT_DIR, cls)))
        print(f"  - {cls}: {count} samples")

if __name__ == "__main__":
    prepare_hybrid()
