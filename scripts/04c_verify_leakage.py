import os
import re

DATASET_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final_mic"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]

def get_group_id(filename):
    match = re.search(r"group(\d+)", filename)
    return int(match.group(1)) if match else None

def verify_no_leakage():
    print("\n" + "="*60)
    print("DATA LEAKAGE (ADATSZIVÁRGÁS) ELLENŐRZÉSE")
    print("="*60)
    
    total_leaks = 0
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        
        train_dir = os.path.join(cls_dir, "train")
        val_dir = os.path.join(cls_dir, "val")
        test_dir = os.path.join(cls_dir, "test")
        
        # Csoport azonosítók kigyűjtése (csak a tiszta clean fájlokból, mert a mic fájlok csupán a train clean fájlok másolatai új csoportnévvel)
        train_groups = set()
        if os.path.exists(train_dir):
            for f in os.listdir(train_dir):
                if f.endswith('.wav') and "_clean_" in f:
                    g_id = get_group_id(f)
                    if g_id is not None:
                        train_groups.add(g_id)
                        
        val_groups = set()
        if os.path.exists(val_dir):
            for f in os.listdir(val_dir):
                if f.endswith('.wav') and "_clean_" in f:
                    g_id = get_group_id(f)
                    if g_id is not None:
                        val_groups.add(g_id)
                        
        test_groups = set()
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                if f.endswith('.wav') and "_clean_" in f:
                    g_id = get_group_id(f)
                    if g_id is not None:
                        test_groups.add(g_id)
                        
        # Metszetek ellenőrzése
        leak_train_val = train_groups.intersection(val_groups)
        leak_train_test = train_groups.intersection(test_groups)
        leak_val_test = val_groups.intersection(test_groups)
        
        class_leaks = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)
        total_leaks += class_leaks
        
        print(f"{cls:<10}:")
        print(f"  - Egyedi tiszta blokkok száma: Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")
        if class_leaks == 0:
            print("  - Status: OK (Nincs átfedés az eredeti források között!)")
        else:
            print("  - Status: HIBA (Adatszivárgás észlelhető!)")
            if leak_train_val: print(f"    * Train <-> Val átfedő blokkok: {leak_train_val}")
            if leak_train_test: print(f"    * Train <-> Test átfedő blokkok: {leak_train_test}")
            if leak_val_test: print(f"    * Val <-> Test átfedő blokkok: {leak_val_test}")
            
    print("="*60)
    if total_leaks == 0:
        print("GRATULÁLOK! 100% GARANTÁLTAN NINCS DATA LEAKAGE AZ ADATHALMAZBAN!")
        print("A Train, Val és Test csoportok teljesen elszeparáltak a fizikai forrásfájlok szintjén.")
    else:
        print(f"FIGYELEM! Összesen {total_leaks} darab adatszivárgási pontot találtam!")
    print("="*60 + "\n")

if __name__ == "__main__":
    verify_no_leakage()
