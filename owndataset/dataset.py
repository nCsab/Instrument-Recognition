import os
import sys
import glob
import librosa
import soundfile as sf

SR = 16000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "instruments", "guitar")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    if len(sys.argv) != 3:
        sys.exit('Usage: python3 dataset.py "filename" start_sec\nExample: python3 dataset.py "piano_piece" 10')

    file_arg = sys.argv[1]
    extensions = ["", ".wav", ".mp3", ".m4a", ".flac", ".ogg"]
    f_path = next((file_arg + ext for ext in extensions if os.path.exists(file_arg + ext)), None)
    
    if not f_path:
        sys.exit(f"Error: File '{file_arg}' not found.")

    try:
        start_s = float(sys.argv[2])
    except ValueError:
        sys.exit("Error: start_sec must be a number.")

    print(f"Loading: {f_path}")
    y, _ = librosa.load(f_path, sr=SR)
    start, end = int(start_s * SR), int((start_s + 5.0) * SR)

    if end > len(y):
        sys.exit(f"Warning: Requested segment ({start_s}-{start_s+5.0}s) exceeds file length, skipping.")

    idx = max([int(os.path.basename(f).split('_')[1].split('.')[0]) for f in glob.glob(os.path.join(OUTPUT_DIR, "block_*.wav"))] + [0]) + 1
    out_path = os.path.join(OUTPUT_DIR, f"block_{idx:03d}.wav")

    sf.write(out_path, y[start:end], SR)
    print(f"Saved 5s block: {out_path}")

    print(f"Cleaning up: removing {f_path}...")
    os.remove(f_path)
    print("Done.")

if __name__ == "__main__":
    main()
