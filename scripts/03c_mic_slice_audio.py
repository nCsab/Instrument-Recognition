import os
import shutil
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
OUTPUT_BASE_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000
SLICE_DURATION = 1.0


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
    file_path = os.path.join(INPUT_DIR, f"{cat}_recorded.wav")
    if not os.path.exists(file_path):
        print(f"Not found: {file_path}")
        return

    print(f"Slicing: {cat}")
    y, _ = librosa.load(file_path, sr=SR)

    beep_end = find_beep_end(y, SR)
    print(f"  Beep end: {beep_end / SR:.2f}s")

    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{cat}_mic_1sec")
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


if __name__ == "__main__":
    for cat in CATEGORIES:
        slice_audio(cat)
    print("\nDone.")
