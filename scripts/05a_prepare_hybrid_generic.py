import os
import shutil
import random
import csv

ROOT_DIR = "/Volumes/Kingston XS1000 Media/project/dataset"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset"
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


def _medley_files(medley_dir, instrument_keyword):
    results = []
    csv_file = os.path.join(medley_dir, "Medley-solos-DB_metadata.csv")
    if not os.path.exists(csv_file): return results
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if instrument_keyword in row.get('instrument', '').lower():
                filename = f"Medley-solos-DB_{row['subset']}-{row['instrument_id']}_{row['uuid4']}.wav"
                path = os.path.join(medley_dir, filename)
                if os.path.exists(path): results.append(path)
    return results


def collect_guitar_sources():
    sources = []
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData/gel")))
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData/gac")))
    sources.extend(_medley_files(os.path.join(ROOT_DIR, "Medley-solos-DB"), "guitar"))
    nsynth_audio = os.path.join(ROOT_DIR, "nsynth-train/audio")
    if os.path.exists(nsynth_audio):
        sources.extend([os.path.join(nsynth_audio, f) for f in os.listdir(nsynth_audio) if f.startswith("guitar_") and f.endswith(".wav")])
    return sources


def collect_piano_sources():
    sources = []
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData/pia")))
    sources.extend(_medley_files(os.path.join(ROOT_DIR, "Medley-solos-DB"), "piano"))
    nsynth_audio = os.path.join(ROOT_DIR, "nsynth-train/audio")
    if os.path.exists(nsynth_audio):
        sources.extend([os.path.join(nsynth_audio, f) for f in os.listdir(nsynth_audio) if (f.startswith("keyboard_") or f.startswith("piano_")) and f.endswith(".wav")])
    return sources


def collect_other_sources():
    sources = []
    for d in ["cel", "cla", "flu", "org", "sax", "tru", "vio", "voi"]:
        sources.extend(get_audio_files(os.path.join(ROOT_DIR, "IRMAS-TrainingData", d)))
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "TinySOL")))
    sources.extend(get_audio_files(os.path.join(ROOT_DIR, "good-sounds")))
    nsynth_audio = os.path.join(ROOT_DIR, "nsynth-train/audio")
    if os.path.exists(nsynth_audio):
        other_files = [os.path.join(nsynth_audio, f) for f in os.listdir(nsynth_audio) if f.endswith(".wav") and not (f.startswith("guitar_") or f.startswith("keyboard_") or f.startswith("piano_"))]
        sources.extend(random.sample(other_files, min(len(other_files), 2000)))
    return sources


def collect_noise_sources():
    return get_audio_files(os.path.join(ROOT_DIR, "ESC-50-master/audio"))


def prepare_hybrid():
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    for cls in CLASSES: os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
    
    funcs = {"guitar": collect_guitar_sources, "piano": collect_piano_sources, "other": collect_other_sources, "noise": collect_noise_sources}
    for cls, func in funcs.items():
        all_sources = func()
        selected = random.sample(all_sources, min(len(all_sources), SAMPLES_PER_CLASS))
        for i, src in enumerate(selected):
            shutil.copy(src, os.path.join(OUTPUT_DIR, cls, f"{cls}_{i:04d}{os.path.splitext(src)[1]}"))
    print("\nHybrid dataset created.")


if __name__ == "__main__":
    prepare_hybrid()
