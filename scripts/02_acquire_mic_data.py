import os
import re
import shutil
import time

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.join(PROJECT_ROOT, "dataset_clean")
RECORD_DIR = os.path.join(PROJECT_ROOT, "owndataset", "record")

CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SPLITS = ["train", "val", "test"]
SR = 16000
CLIP_SECONDS = 1.0
GAP_SECONDS = 0.2


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def record_path(split, *parts):
    return os.path.join(RECORD_DIR, split, *parts)


def selected_splits():
    value = input("Split (train/val/test/all): ").strip().lower()
    return [value] if value in SPLITS else SPLITS


def beep(seconds=1.0, freq=440.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def clean_files(category, split):
    folder = os.path.join(BASE_DIR, category, split)
    if not os.path.exists(folder):
        return []
    files = [name for name in os.listdir(folder) if name.endswith(".wav") and "clean" in name]
    return [os.path.join(folder, name) for name in sorted(files, key=natural_key)]


def prepare_files():
    print("\nPreparing playback files...")
    for split in SPLITS:
        playback_dir = record_path(split, "playback")
        os.makedirs(playback_dir, exist_ok=True)
        print(f"\nSplit: {split}")

        for category in CATEGORIES:
            files = clean_files(category, split)
            if not files:
                print(f"  {category}: no files")
                continue

            audio = [beep(), np.zeros(SR)]
            for file_path in files:
                data, _ = sf.read(file_path)
                audio += [data, np.zeros(int(SR * GAP_SECONDS))]

            output = os.path.join(playback_dir, f"{category}_for_mic.wav")
            sf.write(output, np.concatenate(audio), SR)
            print(f"  {category}: {len(files)} clips -> {output}")


def record_one(split, category):
    playback_file = record_path(split, "playback", f"{category}_for_mic.wav")
    recorded_dir = record_path(split, "recorded")
    os.makedirs(recorded_dir, exist_ok=True)

    duration = 300.0
    if os.path.exists(playback_file):
        info = sf.info(playback_file)
        duration = (info.frames / info.samplerate) + 5.0

    input(f"\nPress Enter to record {split.upper()} / {category.upper()}")
    print(f"Recording for {duration:.1f}s...")
    recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)

    start_time = time.time()
    try:
        while sd.get_stream().active:
            elapsed = time.time() - start_time
            print(f"\r  {int(elapsed)//60:02d}:{int(elapsed)%60:02d} / {int(duration)//60:02d}:{int(duration)%60:02d}", end="", flush=True)
            time.sleep(1)
        print()
    except KeyboardInterrupt:
        sd.stop()
        print("\nRecording stopped early.")

    output = os.path.join(recorded_dir, f"{category}_recorded.wav")
    sf.write(output, recording, SR)
    print(f"Saved: {output}")


def record_mic():
    print("\nInteractive microphone recorder")
    for split in selected_splits():
        for category in CATEGORIES:
            record_one(split, category)


def find_beep_end(audio, freq=440.0):
    spectrum = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=2048)
    energy = spectrum[np.argmin(np.abs(freqs - freq))]
    if np.max(energy) == 0:
        return 0

    energy = energy / np.max(energy)
    above = np.where(energy > 0.5)[0]
    if len(above) == 0:
        return 0

    below = np.where(energy[above[0]:] < 0.2)[0]
    end_frame = above[0] + below[0] if len(below) else len(energy) - 1
    return end_frame * 512


def slice_audio(split, category):
    source = record_path(split, "recorded", f"{category}_recorded.wav")
    if not os.path.exists(source):
        print(f"Not found: {source}")
        return

    audio, _ = librosa.load(source, sr=SR)
    start = find_beep_end(audio) + SR
    step = int((CLIP_SECONDS + GAP_SECONDS) * SR)
    clip_len = int(CLIP_SECONDS * SR)
    output_dir = record_path(split, "slices", f"{category}_mic_1sec")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    count = 0
    while start + clip_len <= len(audio):
        clip = audio[start:start + clip_len]
        sf.write(os.path.join(output_dir, f"{category}_mic_{count:04d}.wav"), clip, SR)
        start += step
        count += 1
    print(f"{split}/{category}: {count} slices")


def slice_all():
    for split in selected_splits():
        for category in CATEGORIES:
            slice_audio(split, category)


def main():
    print("\nMICROPHONE DATA ACQUISITION PIPELINE\n")
    print("1. Prepare playback files")
    print("2. Record microphone audio")
    print("3. Slice recorded audio")
    choice = input("Select an option (or q): ").strip().lower()

    if choice == "1":
        prepare_files()
    elif choice == "2":
        record_mic()
    elif choice == "3":
        slice_all()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
