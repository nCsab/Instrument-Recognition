import os
import shutil
import librosa
import soundfile as sf
import numpy as np

INPUT_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
OUTPUT_BASE_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/recorded_from_mic"
CATEGORIES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000
SLICE_DURATION = 1.0


def find_beep_end(y, sr, freq=440.0):
    # A 440 Hz-es sípszó VÉGÉNEK megkeresése STFT (spektrogram) segítségével.
    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Megkeressük a 440 Hz-hez legközelebbi frekvenciasávot
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bin_idx = np.argmin(np.abs(freqs - freq))
    
    # Kinyerjük a 440 Hz-es frekvenciasáv energiáját az idő függvényében
    energy = D[bin_idx, :]
    
    if np.max(energy) == 0: 
        return 0
        
    energy = energy / np.max(energy)
    
    # Megkeressük azokat a részeket, ahol hangos a 440 Hz-es hang
    above_thresh = np.where(energy > 0.5)[0]
    if len(above_thresh) == 0:
        return 0
        
    start_frame = above_thresh[0]
    
    # Megkeressük, hol esik le az energia 0.2 alá a kezdőpont után (ez a síp VÉGE)
    below_thresh = np.where(energy[start_frame:] < 0.2)[0]
    if len(below_thresh) == 0:
        end_frame = len(energy) - 1
    else:
        end_frame = start_frame + below_thresh[0]
        
    return end_frame * hop_length


def slice_audio(cat):
    file_path = os.path.join(INPUT_DIR, f"{cat}_recorded.wav")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Slicing: {cat}")
    y, _ = librosa.load(file_path, sr=SR)
    
    beep_end_sample = find_beep_end(y, SR)
    print(f"  Sync beep end found at: {beep_end_sample / SR:.2f}s")

    output_dir = os.path.join(OUTPUT_BASE_DIR, f"{cat}_mic_1sec")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Régi szeletek törlése
        print(f"  Régi mappa törölve: {output_dir}")
    os.makedirs(output_dir)

    # A minta hossza 1.0s
    slice_samples = int(SLICE_DURATION * SR)
    
    # A lépésköz (step) 1.2s: 1.0s hang + 0.2s szünet a generáló script alapján
    step_samples = int(1.2 * SR)
    
    # Az első minta pontosan 1.0 másodperccel a sípszó vége után kezdődik
    start_sample = beep_end_sample + int(1.0 * SR)
    count = 0
    
    while start_sample + slice_samples <= len(y):
        chunk = y[start_sample : start_sample + slice_samples]
        
        # Ha nagyon halk (háttérzaj szintje), akkor átugorjuk
        if np.max(np.abs(chunk)) < 0.005:
            start_sample += step_samples
            continue
            
        out_name = f"{cat}_mic_{count:04d}.wav"
        sf.write(os.path.join(output_dir, out_name), chunk, SR)
        count += 1
        
        # Lépünk a következő mintára (1.2 másodperc múlva)
        start_sample += step_samples

    print(f"  Done: {count} slices saved.")


if __name__ == "__main__":
    for cat in CATEGORIES:
        slice_audio(cat)
    print("\nSlicing complete.")
