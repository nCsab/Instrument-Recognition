import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import time

TO_RECORD_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/to_record"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Interactive microphone recorder")
print("-" * 40)

for cat in CATEGORIES:
    input_wav = os.path.join(TO_RECORD_DIR, f"{cat}_for_mic.wav")
    if not os.path.exists(input_wav):
        print(f"Skipped: {input_wav} not found.")
        continue

    info = sf.info(input_wav)
    duration = info.duration + 5

    print(f"\nCategory: {cat.upper()}")
    print(f"Prepare '{cat}_for_mic.wav' on your phone.")
    print(f"Recording duration: {int(duration)} seconds.")
    input("Press ENTER to start recording...")

    print("Recording started. Start playback on your phone now.")

    try:
        recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)

        for i in range(int(duration), 0, -1):
            print(f"\rRemaining: {i:3d}s  ", end="")
            time.sleep(1)

        sd.wait()

        out_path = os.path.join(OUTPUT_DIR, f"{cat}_recorded.wav")
        sf.write(out_path, recording, SR)
        print(f"\nSaved: {out_path}")

    except Exception as e:
        print(f"\nError during recording: {e}")

print("\nAll categories recorded.")
print("Next step: run 04_mic_slice_audio.py")
