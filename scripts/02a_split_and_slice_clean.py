import os
import random
import shutil
import librosa
import soundfile as sf
import numpy as np

# --- KONFIGURÁCIÓ ---
OWNDATASET_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]
SR = 16000
BLOCK_DURATION = 5.0
CLIP_DURATION = 1.0

# Split arányok blokkokban
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def get_audio_files(directory):
    if not os.path.exists(directory): return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

def slice_and_save(block_path, dest_dir, prefix, group_id):
    y, _ = librosa.load(block_path, sr=SR)
    clip_samples = int(CLIP_DURATION * SR)
    num_clips = min(len(y) // clip_samples, int(BLOCK_DURATION / CLIP_DURATION))
    
    saved_paths = []
    for i in range(num_clips):
        start = i * clip_samples
        end = start + clip_samples
        clip = y[start:end]
        
        if np.max(np.abs(clip)) < 0.01: # Skip silent clips
            continue
            
        out_name = f"{prefix}_group{group_id}_clip{i:02d}.wav"
        out_path = os.path.join(dest_dir, out_name)
        sf.write(out_path, clip, SR)
        saved_paths.append(out_path)
    return saved_paths

def process_class(cls):
    print(f"\nFeldolgozás: {cls}...")
    
    # 1. Clean blokkok beolvasása
    clean_blocks_dir = os.path.join(OWNDATASET_DIR, cls, f"{cls}_5sec")
    clean_blocks = get_audio_files(clean_blocks_dir)
    clean_blocks.sort() # Fontos, hogy determinisztikus legyen
    
    if not clean_blocks:
        print(f"Hiba: nincsenek 5s blokkok a {clean_blocks_dir} mappában!")
        return

    # Csoportosítás altípus (prefix) alapján
    blocks_by_prefix = {}
    for block in clean_blocks:
        basename = os.path.basename(block)
        parts = basename.split("_block")
        if len(parts) > 1:
            prefix = parts[0]
        else:
            prefix = cls
            
        if prefix not in blocks_by_prefix:
            blocks_by_prefix[prefix] = []
        blocks_by_prefix[prefix].append(block)

    train_clean_blocks = []
    val_clean_blocks = []
    test_clean_blocks = []
    
    # Rétegzett (stratified) szétosztás altípusonként
    random.seed(42)
    for prefix, blocks in blocks_by_prefix.items():
        random.shuffle(blocks)
        
        num_blocks = len(blocks)
        num_train = int(num_blocks * TRAIN_RATIO)
        num_val = int(num_blocks * VAL_RATIO)
        
        # Elosztás
        train_clean_blocks.extend(blocks[:num_train])
        val_clean_blocks.extend(blocks[num_train:num_train + num_val])
        test_clean_blocks.extend(blocks[num_train + num_val:])
        
        print(f"  - {prefix}: {num_blocks} blokk -> Train={num_train}, Val={num_val}, Test={num_blocks - num_train - num_val}")
    
    num_clean = len(clean_blocks)
    print(f"Összes clean blokk ({num_clean}): Train={len(train_clean_blocks)}, Val={len(val_clean_blocks)}, Test={len(test_clean_blocks)}")

    # Mappák létrehozása a splithez
    cls_out_dir = os.path.join(OUTPUT_DIR, cls)
    if os.path.exists(cls_out_dir):
        shutil.rmtree(cls_out_dir)
    
    train_dir = os.path.join(cls_out_dir, "train")
    val_dir = os.path.join(cls_out_dir, "val")
    test_dir = os.path.join(cls_out_dir, "test")
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Clean blokkok szeletelése és mentése a megfelelő mappába
    group_idx = 1
    for block in train_clean_blocks:
        slice_and_save(block, train_dir, f"{cls}_clean", group_idx)
        group_idx += 1
        
    for block in val_clean_blocks:
        slice_and_save(block, val_dir, f"{cls}_clean", group_idx)
        group_idx += 1
        
    for block in test_clean_blocks:
        slice_and_save(block, test_dir, f"{cls}_clean", group_idx)
        group_idx += 1

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Takarítás: {OUTPUT_DIR}...")
        for cls in CLASSES:
            d = os.path.join(OUTPUT_DIR, cls)
            if os.path.exists(d): shutil.rmtree(d)
            
    for cls in CLASSES:
        process_class(cls)
        
    print("\nAdathalmaz szétválasztása és szeletelése sikeresen befejeződött!")
    print("Minden osztályban létrejött a 'train', 'val' és 'test' almappa az eredeti tiszta hangokkal.")

if __name__ == "__main__":
    main()
