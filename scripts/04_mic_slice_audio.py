import os
import soundfile as sf
import numpy as np

MANUAL_OFFSET = 0.0

INPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
BASE_OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]

SR = 16000
CLIP_LEN = SR
GAP_LEN = int(SR * 0.2)
PERIOD = CLIP_LEN + GAP_LEN


def find_beep_end(data, sr, search_seconds=10.0):
    search_limit = int(sr * search_seconds)
    search_data = data[:search_limit]

    if len(search_data) == 0:
        return None

    window_size = int(sr * 0.05)
    energies = []
    for i in range(0, len(search_data) - window_size, window_size):
        chunk = search_data[i:i + window_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        energies.append(rms)

    if not energies:
        return None

    peak_energy = max(energies)

    in_beep = False
    beep_end_window = None

    for i, e in enumerate(energies):
        if e > peak_energy * 0.4:
            in_beep = True
        elif in_beep and e < peak_energy * 0.1:
            beep_end_window = i
            break

    if beep_end_window is None:
        return None

    beep_end_sample = beep_end_window * window_size
    data_start = beep_end_sample + sr + int(MANUAL_OFFSET * sr)

    return max(0, data_start)


if not os.path.exists(INPUT_DIR):
    os.makedirs(INPUT_DIR)
    print(f"Created directory: {INPUT_DIR}")
    print("Place your recordings here as '{category}_recorded.wav'")

for cat in CATEGORIES:
    input_file = os.path.join(INPUT_DIR, f"{cat}_recorded.wav")
    if not os.path.exists(input_file):
        continue

    output_dir = os.path.join(BASE_OUTPUT_DIR, cat, f"{cat}_mic_1sec")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for old_file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, old_file))

    print(f"Processing {cat}...")
    data, sr = sf.read(input_file)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    start_index = find_beep_end(data, sr)

    if start_index is None:
        print(f"  Error: no sync signal found in {cat}")
        continue

    print(f"  Synced. Data starts at {start_index/sr:.2f}s")

    count = 0
    while start_index + CLIP_LEN <= len(data):
        clip = data[start_index : start_index + CLIP_LEN]

        if np.max(np.abs(clip)) > 0.001:
            out_name = f"{cat}_mic_{count:03d}.wav"
            sf.write(os.path.join(output_dir, out_name), clip, sr)
            count += 1

        start_index += PERIOD

    print(f"  Done: {count} slices saved.")

print("\nSlicing complete. Next step: run 05_assemble_dataset.py")
