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
    input(f"\nPress Enter to start recording: {cat.upper()}")
    
    print(f"Recording {cat}...")
    try:
        duration = 300
        recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)
        
        print("Recording... Press Ctrl+C to stop early and save.")
        try:
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("\nRecording stopped.")

        out_path = os.path.join(OUTPUT_DIR, f"{cat}_recorded.wav")
        sf.write(out_path, recording, SR)
        print(f"\nSaved: {out_path}")

    except Exception as e:
        print(f"\nError during recording: {e}")

print("\nAll categories recorded.")
