import os
import re
import shutil

DATASET_DIR = "/Volumes/Kingston XS1000 Media/project/hybrid_dataset_own_final"
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass"]

def get_group_id(filename):
    # Kinyeri a csoport azonosítót pl. "guitar_clean_group15_clip02.wav" -> "group15"
    match = re.search(r"group\d+", filename)
    return match.group(0) if match else None

def balance_class(cls):
    cls_dir = os.path.join(DATASET_DIR, cls)
    val_dir = os.path.join(cls_dir, "val")
    test_dir = os.path.join(cls_dir, "test")

    if not os.path.exists(val_dir) or not os.path.exists(test_dir):
        print(f"Hiba: {cls} esetében nem található a val vagy test mappa.")
        return

    # Fájlok beolvasása
    val_files = [f for f in os.listdir(val_dir) if f.endswith('.wav')]
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]

    initial_val_count = len(val_files)
    initial_test_count = len(test_files)

    # Blokkok szerint csoportosítás
    # Kulcs: group_id, Érték: { "current_dir": dir_path, "files": [filenames] }
    blocks = {}

    for f in val_files:
        group_id = get_group_id(f)
        if not group_id:
            continue
        if group_id not in blocks:
            blocks[group_id] = {"current_dir": val_dir, "files": []}
        blocks[group_id]["files"].append(f)

    for f in test_files:
        group_id = get_group_id(f)
        if not group_id:
            continue
        if group_id not in blocks:
            blocks[group_id] = {"current_dir": test_dir, "files": []}
        blocks[group_id]["files"].append(f)

    # Mohó elosztás a szeletszámok kiegyenlítésére
    # Rendezzük a blokkokat csökkenő sorrendbe a szeletek száma szerint
    sorted_blocks = sorted(blocks.items(), key=lambda x: len(x[1]["files"]), reverse=True)

    new_val_blocks = []
    new_test_blocks = []
    val_clip_count = 0
    test_clip_count = 0

    for group_id, info in sorted_blocks:
        clip_count = len(info["files"])
        # Mindig oda tesszük, ahol kevesebb szelet van jelenleg
        if val_clip_count <= test_clip_count:
            new_val_blocks.append((group_id, info))
            val_clip_count += clip_count
        else:
            new_test_blocks.append((group_id, info))
            test_clip_count += clip_count

    # Fájlok átmozgatása a döntés alapján
    moved_count = 0
    
    # Val-ba helyezendő fájlok átrakása, ha jelenleg a test-ben vannak
    for group_id, info in new_val_blocks:
        if info["current_dir"] == test_dir:
            for f in info["files"]:
                src = os.path.join(test_dir, f)
                dst = os.path.join(val_dir, f)
                shutil.move(src, dst)
                moved_count += 1

    # Test-be helyezendő fájlok átrakása, ha jelenleg a val-ban vannak
    for group_id, info in new_test_blocks:
        if info["current_dir"] == val_dir:
            for f in info["files"]:
                src = os.path.join(val_dir, f)
                dst = os.path.join(test_dir, f)
                shutil.move(src, dst)
                moved_count += 1

    print(f"{cls:<10} | Eredeti: Val={initial_val_count:<3} Test={initial_test_count:<3} | Új: Val={val_clip_count:<3} Test={test_clip_count:<3} | Mozgatott fájlok: {moved_count}")

def main():
    print("\n" + "="*70)
    print("VAL ÉS TEST HALMAZOK KIEGYENLÍTÉSE (BLOKK SZINTEN)")
    print("="*70)
    
    for cls in CLASSES:
        balance_class(cls)
        
    print("="*70)
    print("Kiegyenlítés sikeresen befejeződött!\n")

if __name__ == "__main__":
    main()
