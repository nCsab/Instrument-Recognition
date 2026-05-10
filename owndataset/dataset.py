import librosa
import soundfile as sf
import os
import subprocess
import re
import sys

# --- KONFIGURÁCIÓ ---
SR = 16000
OUTPUT_DIR_1S = "reed_1sec" # A végleges 1mp-es szeletek mappája
OUTPUT_DIR_5S = "reed_5sec" # Az ideiglenes 5mp-es blokkok mappája

# Mappák létrehozása
for d in [OUTPUT_DIR_1S, OUTPUT_DIR_5S]:
    if not os.path.exists(d):
        os.makedirs(d)

def find_file_with_extensions(filename):
    """Megpróbálja megtalálni a fájlt kiterjesztés nélkül is."""
    if not filename:
        return None
    if os.path.exists(filename):
        return filename
    
    extensions = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    for ext in extensions:
        if os.path.exists(filename + ext):
            return filename + ext
    return None

def get_next_index(directory, prefix):
    """Megkeresi a mappában lévő utolsó indexet és visszaadja a következőt."""
    if not os.path.exists(directory):
        return 1
    files = os.listdir(directory)
    indices = []
    # Keressük a prefix_001.wav formátumú fájlokat
    pattern = re.compile(rf"{prefix}_(\d+)\.wav")
    for f in files:
        match = pattern.match(f)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 1

def save_block_and_slice(y, sr, start_s, length_s):
    start = int(start_s * sr)
    end = start + int(length_s * sr)
    
    if end > len(y):
        print(f"FIGYELEM: A kért szakasz ({start_s}-{start_s+length_s}s) túlnyúlik a fájlon. Kihagyás.")
        return

    block = y[start:end]
    
    # 1. 5 mp-es blokk mentése sorszámmal (saxophone_block előtaggal)
    block_idx = get_next_index(OUTPUT_DIR_5S, "saxophone_block")
    block_name = f"saxophone_block_{block_idx:03d}.wav"
    block_path = os.path.join(OUTPUT_DIR_5S, block_name)
    sf.write(block_path, block, sr)
    print(f"Mentett 5mp blokk: {block_path}")

    # 2. 1 mp-es szeletelés sorszámmal (saxophone előtaggal)
    sec = sr
    num_clips = len(block) // sec
    next_idx = get_next_index(OUTPUT_DIR_1S, "saxophone")

    for i in range(num_clips):
        s = i * sec
        e = s + sec
        clip = block[s:e]
        out_name = f"saxophone_{next_idx:03d}.wav"
        out_path = os.path.join(OUTPUT_DIR_1S, out_name)
        sf.write(out_path, clip, sr)
        print(f"  -> Szelet mentve: {out_name}")
        next_idx += 1

if __name__ == "__main__":
    # Paraméter ellenőrzése
    if len(sys.argv) < 3:
        print("\nHASZNÁLAT: python3 dataset.py \"fájlnév\" kezdőmp1 kezdőmp2 ...")
        print("Példa: python3 dataset.py \"zongora_darab\" 10 45 120")
        sys.exit(1)

    input_arg = sys.argv[1]
    # A 2. indextől kezdve minden további paraméter egy kezdőidőpont (mp)
    try:
        start_times = [float(arg) for arg in sys.argv[2:]]
    except ValueError:
        print("\nHIBA: Az időpontoknak számoknak kell lenniük!")
        sys.exit(1)

    work_file = find_file_with_extensions(input_arg)
    
    if work_file:
        print(f"Fájl betöltése: {work_file}")
        y, sr = librosa.load(work_file, sr=SR)
        
        # Szeletelés a megadott időpontok alapján
        for start_s in start_times:
            save_block_and_slice(y, sr, start_s, 5.0)
        
        # Takarítás: a bemeneti fájl törlése
        print(f"\nTakarítás: {work_file} törlése...")
        os.remove(work_file)
        print("Minden művelet kész.")
    else:
        print(f"\nHIBA: A(z) '{input_arg}' fájl nem található.")
        print("TIPP: Ellenőrizd a fájlnevet az owndataset mappában!")
