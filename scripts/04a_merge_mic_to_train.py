import os
import shutil

# --- KONFIGURÁCIÓ ---
OWNDATASET_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset"
SOURCE_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final_mic"
MIC_DIR = os.path.join(OWNDATASET_DIR, "recorded_from_mic")
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]

def get_audio_files(directory):
    if not os.path.exists(directory): return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

def merge_class(cls):
    print(f"\nFeldolgozás: {cls}...")
    
    train_dir = os.path.join(OUTPUT_DIR, cls, "train")
    if not os.path.exists(train_dir):
        print(f"Hiba: {train_dir} nem létezik.")
        return

    # Megkeressük az utolsó group indexet a train mappában
    train_files = get_audio_files(train_dir)
    max_group = 0
    for f in train_files:
        basename = os.path.basename(f)
        if "_group" in basename:
            try:
                group_num = int(basename.split("_group")[1].split("_")[0])
                if group_num > max_group:
                    max_group = group_num
            except:
                pass
                
    group_idx = max_group + 1

    # Mic klipek beolvasása és KIZÁRÓLAG a train mappába másolása
    mic_clips_dir = os.path.join(MIC_DIR, f"{cls}_mic_1sec")
    mic_clips = get_audio_files(mic_clips_dir)
    
    if mic_clips:
        print(f"Mic klipek ({len(mic_clips)}): Mind a TRAIN halmazba másolva.")
        for i, mic_clip in enumerate(mic_clips):
            # Csoportosítjuk őket 5-ösével
            mic_group = group_idx + (i // 5)
            out_name = f"{cls}_mic_group{mic_group}_clip{i%5:02d}.wav"
            shutil.copy(mic_clip, os.path.join(train_dir, out_name))
    else:
        print(f"Figyelem: nincsenek mic klipek a {mic_clips_dir} mappában. Vedd fel őket a 04 és 05 scriptekkel!")

def balance_noise():
    print("\n--- Zaj (noise) osztály fizikai kiegyenlítése a lemezen ---")
    
    project_dir = "/Volumes/Kingston XS1000 Media/project"
    noise_pool = os.path.join(project_dir, "noise_train_pool")
    clean_noise_train = os.path.join(SOURCE_DIR, "noise", "train")
    mic_noise_train = os.path.join(OUTPUT_DIR, "noise", "train")
    
    import random
    os.makedirs(noise_pool, exist_ok=True)
    
    # 1. Pool feltöltése az első alkalommal a clean noise train mappából
    if os.path.exists(clean_noise_train):
        clean_noise_files = [f for f in os.listdir(clean_noise_train) if f.endswith(".wav")]
        pool_files = [f for f in os.listdir(noise_pool) if f.endswith(".wav")]
        
        # Ha a pool-ban kevesebb fájl van, mint a jelenlegi clean noise train-ben (pl. először futtatjuk)
        if len(pool_files) < len(clean_noise_files):
            print(f"Zajfájl pool másolása ({len(clean_noise_files)} fájl) ide: {noise_pool}...")
            for f in clean_noise_files:
                src_path = os.path.join(clean_noise_train, f)
                dst_path = os.path.join(noise_pool, f)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
    
    pool_files = [f for f in os.listdir(noise_pool) if f.endswith(".wav")]
    if not pool_files:
        print("Hiba: A zaj pool üres és a tiszta zaj train mappa sem található!")
        return

    pool_files.sort() # determinisztikus rendezés
    
    # 2. Cél darabszámok kiszámítása a hangszeres train osztályok átlagaként
    # Tiszta adathalmaz hangszeres átlaga:
    clean_counts = []
    for cls in CLASSES:
        cls_train_dir = os.path.join(SOURCE_DIR, cls, "train")
        if os.path.exists(cls_train_dir):
            count = len([f for f in os.listdir(cls_train_dir) if f.endswith(".wav")])
            clean_counts.append(count)
            
    if clean_counts:
        n_clean = int(round(sum(clean_counts) / len(clean_counts)))
    else:
        n_clean = 346 # fallback
        
    # Mikrofonos adathalmaz hangszeres átlaga:
    mic_counts = []
    for cls in CLASSES:
        cls_train_dir = os.path.join(OUTPUT_DIR, cls, "train")
        if os.path.exists(cls_train_dir):
            count = len([f for f in os.listdir(cls_train_dir) if f.endswith(".wav")])
            mic_counts.append(count)
            
    if mic_counts:
        n_mic = int(round(sum(mic_counts) / len(mic_counts)))
    else:
        n_mic = 690 # fallback
        
    n_clean_noise = n_clean * 2
    n_mic_noise = min(len(pool_files), n_mic * 2)

    print(f"Hangszeres train osztályok átlaga -> Tiszta (clean): {n_clean} (Cél zaj: {n_clean_noise}), Mikrofonos (mic): {n_mic} (Cél zaj: {n_mic_noise})")
    
    # 3. Determinisztikus mintavételezés
    random.seed(42)
    shuffled_pool = list(pool_files)
    random.shuffle(shuffled_pool)
    
    selected_mic = shuffled_pool[:n_mic_noise]
    selected_clean = shuffled_pool[:n_clean_noise] # a tiszta a mikrofonos részhalmaza!
    
    # 4. Fájlok másolása a tiszta adathalmaz noise/train mappájába
    if os.path.exists(clean_noise_train):
        print(f"Takarítás és másolás a tiszta zaj train mappába ({n_clean_noise} fájl)...")
        shutil.rmtree(clean_noise_train)
    os.makedirs(clean_noise_train, exist_ok=True)
    for f in selected_clean:
        shutil.copy2(os.path.join(noise_pool, f), os.path.join(clean_noise_train, f))
        
    # 5. Fájlok másolása a mikrofonos adathalmaz noise/train mappájába
    if os.path.exists(mic_noise_train):
        print(f"Takarítás és másolás a mikrofonos zaj train mappába ({n_mic_noise} fájl)...")
        shutil.rmtree(mic_noise_train)
    os.makedirs(mic_noise_train, exist_ok=True)
    for f in selected_mic:
        shutil.copy2(os.path.join(noise_pool, f), os.path.join(mic_noise_train, f))
        
    print("Zajosztály kiegyenlítése sikeresen befejeződött mindkét változatban!")

def main():
    print("--- Mikrofonos felvételek beolvasztása a TRAIN halmazba ---")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Hiba: A forrás mappa nem található: {SOURCE_DIR}")
        return
        
    if os.path.exists(OUTPUT_DIR):
        print(f"Takarítás (korábbi másolat törlése): {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
        
    print(f"Új másolat készítése: {SOURCE_DIR} -> {OUTPUT_DIR}...")
    shutil.copytree(SOURCE_DIR, OUTPUT_DIR)
    
    for cls in CLASSES:
        merge_class(cls)
        
    balance_noise()
    
    print("\nFolyamat befejeződött!")
    print(f"A mikrofonos fájlok sikeresen bekerültek a {OUTPUT_DIR} mappába, és a zaj eloszlása kiegyenlítésre került.")
    print("Az eredeti 'hybrid_dataset_own_final' mappa is frissült a kiegyenlített zaj train adatokkal.")

if __name__ == "__main__":
    main()
