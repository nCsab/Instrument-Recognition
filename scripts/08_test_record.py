import os
import time
import sounddevice as sd
import soundfile as sf
from datetime import datetime

SR = 16000
RECORD_SECONDS = 10
OUTPUT_BASE_DIR = "../model_test"


def record_audio(duration, sr):
    print(f"\nRecording starts in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print(f"Recording ({duration}s)...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)

    for i in range(duration, 0, -1):
        print(f"  Remaining: {i:2d}s", end="\r")
        time.sleep(1)

    sd.wait()
    print("\nRecording complete.")
    return recording


def save_recording(recording, sr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rec_{timestamp}.wav"
    file_path = os.path.join(OUTPUT_BASE_DIR, filename)

    sf.write(file_path, recording, sr)
    print(f"Saved: {file_path}")
    return file_path


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    try:
        audio_data = record_audio(RECORD_SECONDS, SR)
        path = save_recording(audio_data, SR)
        print(f"\nDone: {path}")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
