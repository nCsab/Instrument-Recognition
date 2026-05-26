import os
import random
import librosa
import numpy as np
import soundfile as sf
from utils.feature_utils import extract_log_mel, extract_stft, extract_mfcc, z_score_normalize
from utils.augmentation_utils import apply_macbook_augment

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
SR, PREVIEW_PER_CLASS = 16000, 3

random.seed(42)
np.random.seed(42)

def process_batch_dataset(dataset_path, output_path, aug_samples_path, extract_raw=False, do_augmentation=True):
    os.makedirs(output_path, exist_ok=True)
    if aug_samples_path: os.makedirs(aug_samples_path, exist_ok=True)

    data = {s: {'log_mel': [], 'stft': [], 'mfcc': [], 'raw': [], 'labels': []} for s in ['train', 'val', 'test']}
    if not os.path.exists(dataset_path): return print(f"Error: {dataset_path} not found")

    noise_dir = os.path.join(dataset_path, "noise", "train")
    noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")] if os.path.exists(noise_dir) else []
    saved_previews = {}

    def add_feats(y, subset, lbl):
        y = np.pad(y, (0, max(0, SR - len(y))))[:SR] if len(y) < SR else y[:SR]
        data[subset]['log_mel'].append(extract_log_mel(y))
        data[subset]['stft'].append(extract_stft(y))
        data[subset]['mfcc'].append(z_score_normalize(extract_mfcc(y)))
        if extract_raw: data[subset]['raw'].append(y)
        data[subset]['labels'].append(lbl)
        return y

    for lbl, cls in enumerate(CLASSES):
        for subset in ['train', 'val', 'test']:
            d = os.path.join(dataset_path, cls, subset)
            if not os.path.exists(d): continue
            
            files = [f for f in os.listdir(d) if f.endswith(".wav")]
            random.shuffle(files)
            
            if cls == "noise":
                n_inst = len([l for l in data[subset]['labels'] if l < 6])
                target = n_inst // 6 if n_inst else len(files)
                files = files[:target] if target <= len(files) else files + random.choices(files, k=target - len(files))

            print(f"Processing: {cls} - {subset} ({len(files)} samples)")
            for i, f in enumerate(files):
                try:
                    y, _ = librosa.load(os.path.join(d, f), sr=SR)
                    if len(y) < SR: continue
                    y = add_feats(y, subset, lbl)

                    if do_augmentation and subset == 'train' and cls != 'noise' and "_mic_" not in f:
                        aug_y = apply_macbook_augment(y.copy(), noise_files, noise_path=noise_dir, sr=SR)
                        aug_y = add_feats(aug_y, subset, lbl)
                        
                        if aug_samples_path and saved_previews.get(cls, 0) < PREVIEW_PER_CLASS:
                            sf.write(os.path.join(aug_samples_path, f"{cls}_{saved_previews.get(cls, 0):02d}_ORIGINAL.wav"), y, SR)
                            sf.write(os.path.join(aug_samples_path, f"{cls}_{saved_previews.get(cls, 0):02d}_AUGMENTED.wav"), aug_y, SR)
                            saved_previews[cls] = saved_previews.get(cls, 0) + 1
                except Exception as e: print(f"Error {f}: {e}")

    print("\nSaving features...")
    for s in ['train', 'val', 'test']:
        if not data[s]['labels']: continue
        print(f"\nSubset: {s}")
        for feat in ['log_mel', 'stft', 'mfcc'] + (['raw'] if extract_raw else []):
            X = np.array(data[s][feat])
            if feat != 'raw':
                X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
                if feat != 'mfcc': X = (X - X.min()) / (X.max() - X.min() + 1e-10)
            p = os.path.join(output_path, f"X_{feat}_{s}.npy")
            np.save(p, X); print(f"  {feat}: {X.shape} -> {p}")
            
        lp = os.path.join(output_path, f"y_labels_{s}.npy")
        np.save(lp, np.array(data[s]['labels'])); print(f"  labels: {len(data[s]['labels'])} -> {lp}")

def main():
    b = "/Users/csabanagy/Desktop/project"
    for path, out, aug, do_aug, title in [
        (os.path.join(b, "dataset_clean"), os.path.join(b, "processed_data_clean"), None, False, "CLEAN DATASET (No Augmentation)"),
        (os.path.join(b, "dataset_clean"), os.path.join(b, "processed_data_augmented"), os.path.join(b, "augmented_test_clean"), True, "AUGMENTED DATASET"),
        (os.path.join(b, "dataset_mic"), os.path.join(b, "processed_data_mic"), os.path.join(b, "augmented_test_mic"), True, "MIC DATASET")
    ]:
        if os.path.exists(path):
            print(f"\nFEATURE EXTRACTION - {title}\n")
            process_batch_dataset(path, out, aug, extract_raw=True, do_augmentation=do_aug)

if __name__ == "__main__": main()
