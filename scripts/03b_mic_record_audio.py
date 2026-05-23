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
        playback_file = os.path.join(TO_RECORD_DIR, f"{cat}_for_mic.wav")
        if os.path.exists(playback_file):
            info = sf.info(playback_file)
            playback_duration = info.frames / info.samplerate
            duration = playback_duration + 5.0 # 5 másodperc ráhagyás a biztonság kedvéért
            print(f"Detected playback length: {playback_duration:.1f}s. Auto-recording for {duration:.1f}s...")
        else:
            duration = 300
            print(f"Warning: Playback file not found. Defaulting to {duration}s.")
            
        recording = sd.rec(int(duration * SR), samplerate=SR, channels=1)
        
        print("Recording... Press Ctrl+C to stop early and save.")
        try:
            start_time = time.time()
            while sd.get_stream().active:
                elapsed = time.time() - start_time
                remaining = max(0, duration - elapsed)
                mins_e, secs_e = divmod(int(elapsed), 60)
                mins_r, secs_r = divmod(int(remaining), 60)
                progress = min(elapsed / duration, 1.0)
                bar_len = 30
                filled = int(bar_len * progress)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"\r  [{bar}] {mins_e:02d}:{secs_e:02d} / {int(duration)//60:02d}:{int(duration)%60:02d}  (hátra: {mins_r:02d}:{secs_r:02d})", end="", flush=True)
                time.sleep(1)
            print()  # Új sor a visszaszámláló után
        except KeyboardInterrupt:
            sd.stop()
            print("\n\nRecording stopped early.")

        out_path = os.path.join(OUTPUT_DIR, f"{cat}_recorded.wav")
        sf.write(out_path, recording, SR)
        print(f"\nSaved: {out_path}")

    except Exception as e:
        print(f"\nError during recording: {e}")

print("\nAll categories recorded.")
