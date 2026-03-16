
import os
import numpy as np
import librosa
import random

DATASET_PATH = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset"
OUTPUT_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
CLASSES = ["guitar", "piano", "other", "noise"]

SR = 16000
DURATION = 1.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

def add_noise(audio, noise_path, snr_db=10):
    try:
        noise, _ = librosa.load(noise_path, sr=SR, duration=DURATION)
        if len(noise) < len(audio):
            noise = np.pad(noise, (0, len(audio) - len(noise)))
        else:
            noise = noise[:len(audio)]
        
        p_audio = np.sum(audio ** 2) / len(audio)
        p_noise = np.sum(noise ** 2) / len(noise)
        snr = 10 ** (snr_db / 10)
        scale = np.sqrt(p_audio / (snr * p_noise + 1e-10))
        return audio + scale * noise
    except:
        return audio

def add_reverb(audio, delay_ms=30, decay=0.4):
    try:
        delay_samples = int(SR * (delay_ms / 1000.0))
        taps = [delay_samples, int(delay_samples * 1.5), int(delay_samples * 2.1)]
        out = audio.copy()
        for i, t in enumerate(taps):
            if t < len(audio):
                out[t:] += audio[:-t] * (decay ** (i + 1))
        return out
    except:
        return audio

def add_eq(audio):
    try:
        coef = random.uniform(-0.5, 0.5)
        return librosa.effects.preemphasis(audio, coef=coef)
    except:
        return audio

def extract_log_mel(audio):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def process_dataset():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    features = []
    labels = []

    noise_dir = os.path.join(DATASET_PATH, "noise")
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]

    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATASET_PATH, class_name)
        print(f"Processing class: {class_name}")
        
        files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
        random.shuffle(files)
        files = files[:500]
        
        for idx, f in enumerate(files):
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(files)} files...")
            
            file_path = os.path.join(class_dir, f)
            
            try:
                y, _ = librosa.load(file_path, sr=SR)
                
                samples_per_seg = int(SR * DURATION)
                num_segments = len(y) // samples_per_seg
                
                for i in range(num_segments):
                    start = i * samples_per_seg
                    end = start + samples_per_seg
                    segment = y[start:end]
                    
                    log_mel = extract_log_mel(segment)
                    features.append(log_mel)
                    labels.append(label_idx)
                    
                    if class_name != "noise" and random.random() < 0.5:
                        aug_y = segment.copy()
                        
                        augs_to_apply = random.sample(["noise", "pitch", "gain", "reverb", "eq"], random.randint(1, 2))
                        
                        for choice in augs_to_apply:
                            if choice == "noise" and noise_files:
                                n_file = random.choice(noise_files)
                                aug_y = add_noise(aug_y, n_file, snr_db=random.uniform(10, 25))
                            elif choice == "pitch":
                                aug_y = librosa.effects.pitch_shift(aug_y, sr=SR, n_steps=random.uniform(-1.5, 1.5))
                            elif choice == "gain":
                                aug_y = aug_y * random.uniform(0.7, 1.3)
                            elif choice == "reverb":
                                aug_y = add_reverb(aug_y, delay_ms=random.uniform(20, 50), decay=random.uniform(0.3, 0.5))
                            elif choice == "eq":
                                aug_y = add_eq(aug_y)
                            
                        aug_log_mel = extract_log_mel(aug_y)
                        features.append(aug_log_mel)
                        labels.append(label_idx)
                        
            except Exception as e:
                print(f"Error processing {f}: {e}")

    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nFinished extraction. Shape: {X.shape}")
    
    np.save(os.path.join(OUTPUT_PATH, "X_hybrid.npy"), X)
    np.save(os.path.join(OUTPUT_PATH, "y_hybrid.npy"), y)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_dataset()

