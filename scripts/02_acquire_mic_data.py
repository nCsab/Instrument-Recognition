import os
import random
import time
import shutil
import re
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np

# Config
BASE_DIR = "/Volumes/Kingston XS1000 Media/project/dataset_clean"
TO_RECORD_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/record/to_record"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/record/recorded_from_mic"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000
SLICE_DURATION = 1.0


def create_beep(duration_s, freq=440, sr=16000):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def prepare_files():
    print("\nPreparing reference playback files...")
    os.makedirs(TO_RECORD_DIR, exist_ok=True)

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

        sf.write(os.path.join(TO_RECORD_DIR, f"{cat}_for_mic.wav"), np.concatenate(combined_audio), SR)
        print(f"  {cat}: {len(files)} samples concatenated -> {cat}_for_mic.wav")


def record_mic():
    print("\nInteractive microphone recorder")
    print("-" * 40)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cat in CATEGORIES:
        input(f"\nPress Enter to start recording: {cat.upper()}")
        print(f"Recording {cat}...")

        try:
            playback_file = os.path.join(TO_RECORD_DIR, f"{cat}_for_mic.wav")
            if os.path.exists(playback_file):
                info = sf.info(playback_file)
                playback_duration = info.frames / info.samplerate
                duration = playback_duration + 5.0
                print(f"Playback length: {playback_duration:.1f}s, recording for {duration:.1f}s...")
            else:
                duration = 300
                print(f"Warning: playback file not found, defaulting to {duration}s.")

            recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)

            print("Recording... Press Ctrl+C to stop early.")
            try:
                start_time = time.time()
                while sd.get_stream().active:
                    elapsed = time.time() - start_time
                    remaining = max(0, duration - elapsed)
                    progress = min(elapsed / duration, 1.0)
                    bar_len = 30
                    filled = int(bar_len * progress)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {int(elapsed)//60:02d}:{int(elapsed)%60:02d} / "
                          f"{int(duration)//60:02d}:{int(duration)%60:02d}", end="", flush=True)
                    time.sleep(1)
                print()
            except KeyboardInterrupt:
                sd.stop()
                print("\n\nRecording stopped early.")

            out_path = os.path.join(OUTPUT_DIR, f"{cat}_recorded.wav")
            sf.write(out_path, recording, SR)
            print(f"Saved: {out_path}")

        except Exception as e:
            print(f"\nError: {e}")

    print("\nAll categories recorded.")


def find_beep_end(y, sr, freq=440.0):
    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_idx = np.argmin(np.abs(freqs - freq))
    energy = D[bin_idx, :]

    if np.max(energy) == 0:
        return 0
    energy = energy / np.max(energy)

    above = np.where(energy > 0.5)[0]
    if len(above) == 0:
        return 0

    start_frame = above[0]
    below = np.where(energy[start_frame:] < 0.2)[0]
    end_frame = start_frame + below[0] if len(below) > 0 else len(energy) - 1

    return end_frame * hop_length


def slice_audio(cat):
    file_path = os.path.join(OUTPUT_DIR, f"{cat}_recorded.wav")
    if not os.path.exists(file_path):
        print(f"Not found: {file_path}")
        return

    print(f"Slicing: {cat}")
    y, _ = librosa.load(file_path, sr=SR)

    beep_end = find_beep_end(y, SR)
    print(f"  Beep end: {beep_end / SR:.2f}s")

    output_dir = os.path.join(OUTPUT_DIR, f"{cat}_mic_1sec")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    slice_samples = int(SLICE_DURATION * SR)
    step_samples = int(1.2 * SR)  # 1.0s clip + 0.2s gap
    start = beep_end + int(1.0 * SR)
    count = 0

    while start + slice_samples <= len(y):
        chunk = y[start:start + slice_samples]
        if np.max(np.abs(chunk)) < 0.005:
            start += step_samples
            continue
        sf.write(os.path.join(output_dir, f"{cat}_mic_{count:04d}.wav"), chunk, SR)
        count += 1
        start += step_samples

    print(f"  {count} slices saved.")


def slice_all():
    print("\nSlicing all recorded files...")
    for cat in CATEGORIES:
        slice_audio(cat)
    print("Done.\n")


def main():
    print("==================================================")
    print("MICROPHONE DATA ACQUISITION PIPELINE")
    print("==================================================")
    print("1. Prepare reference files (concatenate clean data)")
    print("2. Record mic audio interactively")
    print("3. Slice recorded audio into 1-second segments")
    print("4. Run all steps sequentially")
    print("q. Quit")
    choice = input("\nSelect an option: ").strip().lower()

    if choice == '1':
        prepare_files()
    elif choice == '2':
        record_mic()
    elif choice == '3':
        slice_all()
    elif choice == '4':
        prepare_files()
        record_mic()
        slice_all()
    elif choice == 'q':
        print("Exiting...")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
