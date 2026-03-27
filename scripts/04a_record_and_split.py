import os
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime

# --- KONFIGURÁCIÓ ---
SR = 16000          # Mintavételezési frekvencia (konzisztens a projekttel)
RECORD_SECONDS = 10 # Teljes felvétel hossza
CHUNKS_SECONDS = 2  # Hány másodperces darabokra vágjuk
OUTPUT_BASE_DIR = "../model_test"

def record_audio(duration, sr):
    print(f"\n🎤 KÉSZÜLJ! A felvétel 3 másodperc múlva indul...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    
    print(f"🔴 FELVÉTEL INDÍTVA ({duration} másodperc)...")
    # Felvétel indítása (mono)
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    
    # Visszaszámlálás a felvétel alatt
    for i in range(duration, 0, -1):
        print(f"  Hátralévő idő: {i:2d} mp", end="\r")
        time.sleep(1)
    
    sd.wait()  # Megvárjuk amíg a felvétel tényleg befejeződik
    print(f"\n✅ Felvétel kész!")
    return recording

def save_recording(recording, sr):
    # Időbélyeg alapú fájlnév
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rec_{timestamp}.wav"
    file_path = os.path.join(OUTPUT_BASE_DIR, filename)
    
    sf.write(file_path, recording, sr)
    print(f"💾 Mentve: {file_path}")
    
    return file_path

def main():
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)
        
    try:
        # 1. Felvétel
        audio_data = record_audio(RECORD_SECONDS, SR)
        
        # 2. Mentés
        path = save_recording(audio_data, SR)
        
        print(f"\n✨ SIKER! A felvétel megtalálható: {path}")
        print("Futtasd a 12_test_batch_predict.py-t az elemzéshez!")
        
    except Exception as e:
        print(f"\n❌ HIBA történt: {e}")
        print("Ellenőrizd, hogy a mikrofon hozzáférés engedélyezve van-e a terminál számára!")

if __name__ == "__main__":
    main()

