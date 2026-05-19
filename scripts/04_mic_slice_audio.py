import os
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
OUTPUT_BASE_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000
SLICE_DURATION = 1.0


def find_beep_end(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    if len(onsets) > 0:
        return librosa.frames_to_samples(onsets[0])
    return 0


def slice_audio(cat):
    file_path = os.path.join(INPUT_DIR, f"{cat}_recorded.wav")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Slicing: {cat}")
    y, _ = librosa.load(file_path, sr=SR)
    
    start_sample = find_beep_end(y, SR)
    print(f"  Sync beep end found at: {start_sample / SR:.2f}s")

    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{cat}_mic_1sec")
    os.makedirs(output_dir, exist_ok=True)

    slice_samples = int(SLICE_DURATION * SR)
    count = 0
    
    for i in range(start_sample, len(y) - slice_samples, slice_samples):
        chunk = y[i : i + slice_samples]
        if np.max(np.abs(chunk)) < 0.01:
            continue
            
        out_name = f"{cat}_mic_{count:04d}.wav"
        sf.write(os.path.join(output_dir, out_name), chunk, SR)
        count += 1

    print(f"  Done: {count} slices saved.")


if __name__ == "__main__":
    for cat in CATEGORIES:
        slice_audio(cat)
    print("\nSlicing complete.")
