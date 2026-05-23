import os
import re
import sys
import librosa
import soundfile as sf

SR = 16000
OUTPUT_DIR_5S = "instruments/brass/brass_5sec"

if not os.path.exists(OUTPUT_DIR_5S):
    os.makedirs(OUTPUT_DIR_5S)


def find_file_with_extensions(filename):
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
    if not os.path.exists(directory):
        return 1
    files = os.listdir(directory)
    indices = []
    pattern = re.compile(rf"{prefix}_(\d+)\.wav")
    for f in files:
        match = pattern.match(f)
        if match:
            indices.append(int(match.group(1)))
    return max(indices) + 1 if indices else 1


def save_block(y, sr, start_s, length_s):
    start = int(start_s * sr)
    end = start + int(length_s * sr)
    
    if end > len(y):
        print(f"Figyelem: A kért szakasz ({start_s}-{start_s+length_s}s) túlnyúlik a fájlon, kihagyás.")
        return

    block = y[start:end]
    block_idx = get_next_index(OUTPUT_DIR_5S, "horn_block")
    block_name = f"horn_block_{block_idx:03d}.wav"
    block_path = os.path.join(OUTPUT_DIR_5S, block_name)
    sf.write(block_path, block, sr)
    print(f"Mentett 5s blokk: {block_path}")


def main():
    if len(sys.argv) < 3:
        print("\nHasználat: python3 dataset.py \"fájlnév\" kezdőmp1 kezdőmp2 ...")
        print("Példa: python3 dataset.py \"zongora_darab\" 10 45 120")
        sys.exit(1)

    input_arg = sys.argv[1]
    try:
        start_times = [float(arg) for arg in sys.argv[2:]]
    except ValueError:
        print("\nHiba: Az időpontoknak számoknak kell lenniük.")
        sys.exit(1)

    work_file = find_file_with_extensions(input_arg)
    if work_file:
        print(f"Fájl betöltése: {work_file}")
        y, sr = librosa.load(work_file, sr=SR)
        for start_s in start_times:
            save_block(y, sr, start_s, 5.0)
        print(f"\nTakarítás: {work_file} törlése...")
        os.remove(work_file)
        print("Minden művelet kész.")
    else:
        print(f"\nHiba: A(z) '{input_arg}' fájl nem található.")


if __name__ == "__main__":
    main()
