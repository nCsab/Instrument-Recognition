
import os
import shutil
import random
import csv
from collections import defaultdict

# --- KONFIGURÁCIÓ ---
ROOT_DIR = "/Volumes/Kingston XS1000 Media/project/dataset"
# Saját gyűjtemények:
OWN_GUITAR_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/guitar_1sec"
OWN_PIANO_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/piano_1sec"
OWN_VOCAL_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/vocal_1sec"
OWN_STRING_DIR = "/Volumes/Kingston XS1000 Media/project/owndataset/string_1sec"

# Az új dataset mappája:
OUTPUT_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"

CLASSES = ["guitar", "piano", "vocal", "string", "noise"]

AUDIO_EXTENSIONS = ('.wav', '.flac')

def get_audio_files(directory):
    audio_files = []
    if not os.path.exists(directory):
        return []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, f))
    return audio_files

def collect_guitar_sources():
    sources = get_audio_files(OWN_GUITAR_DIR)
    print(f"  Saját gitár minták: {len(sources)}")
    return sources

def collect_piano_sources():
    sources = get_audio_files(OWN_PIANO_DIR)
    print(f"  Saját zongora minták: {len(sources)}")
    return sources

def collect_vocal_sources():
    sources = get_audio_files(OWN_VOCAL_DIR)
    print(f"  Saját ének(vocal) minták: {len(sources)}")
    return sources

def collect_string_sources():
    sources = get_audio_files(OWN_STRING_DIR)
    print(f"  Saját vonós(string) minták: {len(sources)}")
    return sources

def collect_noise_sources():
    # ESC-50 (vagy bármelyik zaj készlet)
    sources = get_audio_files(os.path.join(ROOT_DIR, "ESC-50-master/audio"))
    if not sources:
        # Próbáljuk a "noise" mappát is, ha létezik
        sources = get_audio_files(os.path.join(ROOT_DIR, "noise"))
    return sources

def prepare_hybrid():
    print("--- Egyedi Hybrid Dataset készítése (Saját Gitár + Zongora + Vocal + String) ---")
    
    # Kiszámoljuk a mintaszámot a saját adatok maximuma alapján
    own_counts = {
        "guitar": len(get_audio_files(OWN_GUITAR_DIR)),
        "piano": len(get_audio_files(OWN_PIANO_DIR)),
        "vocal": len(get_audio_files(OWN_VOCAL_DIR)),
        "string": len(get_audio_files(OWN_STRING_DIR))
    }
    
    for cls, count in own_counts.items():
        print(f"  {cls}: {count} minta")

    # A cél mintaszám a legnagyobb kategória mérete legyen
    SAMPLES_PER_CLASS = max(own_counts.values())
    if SAMPLES_PER_CLASS == 0:
        print("HIBA: Nem találtam saját mintákat!")
        return
        
    print(f"Cél mintaszám (maximum alapján): {SAMPLES_PER_CLASS}")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Mappa törlése: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    for cls in CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

    collection_funcs = {
        "guitar": collect_guitar_sources,
        "piano": collect_piano_sources,
        "vocal": collect_vocal_sources,
        "string": collect_string_sources,
        "noise": collect_noise_sources
    }

    for cls, func in collection_funcs.items():
        print(f"Források gyűjtése: {cls}...")
        all_sources = func()
        
        print(f"  Elérhető összesen: {len(all_sources)}")
        
        # Ha kevesebb van, mint a cél, mindet visszük. Ha több, mintat veszünk.
        sample_size = min(len(all_sources), SAMPLES_PER_CLASS)
        selected = random.sample(all_sources, sample_size)
        
        print(f"  Másolás ({sample_size} db)...")
        for i, src in enumerate(selected):
            ext = os.path.splitext(src)[1]
            dest = os.path.join(OUTPUT_DIR, cls, f"{cls}_{i:04d}{ext}")
            shutil.copy(src, dest)

    print("\nSIKER! Az egyedi hybrid dataset elkészült:")
    print(OUTPUT_DIR)
    
    print("\nEllenőrzés:")
    for cls in CLASSES:
        count = len(os.listdir(os.path.join(OUTPUT_DIR, cls)))
        print(f"  - {cls}: {count} minta")

if __name__ == "__main__":
    prepare_hybrid()
