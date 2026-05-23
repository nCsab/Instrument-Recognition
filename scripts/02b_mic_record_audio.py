import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import time

TO_RECORD_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/to_record"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Interactive microphone recorder")
print("-" * 40)

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
