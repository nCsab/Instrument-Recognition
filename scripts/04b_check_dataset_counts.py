import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR_CLEAN = os.path.join(PROJECT_DIR, "hybrid_dataset_own_final")
DATASET_DIR_MIC = os.path.join(PROJECT_DIR, "hybrid_dataset_own_final_mic")

def print_dataset_stats(dataset_dir, title_suffix=""):
    if not os.path.exists(dataset_dir):
        return False

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()

    print("\n" + "="*60)
    print(f"DATASET {title_suffix} - MINTÁK SZÁMA (1 MP KLIPEK)")
    print(f"Mappa: {os.path.basename(dataset_dir)}")
    print("="*60)
    print(f"{'Osztály':<15} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Összesen':<7}")
    print("-" * 60)

    total_train = 0
    total_val = 0
    total_test = 0

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        
        train_count = 0
        val_count = 0
        test_count = 0
        
        train_dir = os.path.join(cls_dir, "train")
        val_dir = os.path.join(cls_dir, "val")
        test_dir = os.path.join(cls_dir, "test")
        
        if os.path.exists(train_dir):
            train_count = len([f for f in os.listdir(train_dir) if f.endswith('.wav')])
        if os.path.exists(val_dir):
            val_count = len([f for f in os.listdir(val_dir) if f.endswith('.wav')])
        if os.path.exists(test_dir):
            test_count = len([f for f in os.listdir(test_dir) if f.endswith('.wav')])
            
        total_cls = train_count + val_count + test_count
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{cls:<15} | {train_count:<7} | {val_count:<7} | {test_count:<7} | {total_cls:<7}")

    print("-" * 60)
    total_all = total_train + total_val + total_test
    print(f"{'ÖSSZESEN':<15} | {total_train:<7} | {total_val:<7} | {total_test:<7} | {total_all:<7}")
    print("="*60 + "\n")
    return True

def main():
    printed_any = False
    if print_dataset_stats(DATASET_DIR_CLEAN, "TISZTA (CSAK INTERNETES MINTÁK)"):
        printed_any = True
    if print_dataset_stats(DATASET_DIR_MIC, "MIKROFONOS AUGMENTÁLT"):
        printed_any = True
        
    if not printed_any:
        print(f"Hiba: Egyik adathalmaz mappa sem található:\n- {DATASET_DIR_CLEAN}\n- {DATASET_DIR_MIC}")

if __name__ == "__main__":
    main()
