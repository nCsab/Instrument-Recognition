import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PROCESSED_DIR_CLEAN = os.path.join(PROJECT_DIR, "processed_data_clean")
PROCESSED_DIR_MIC = os.path.join(PROJECT_DIR, "processed_data_mic")

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]

def print_feature_stats(processed_dir, title_suffix=""):
    if not os.path.exists(processed_dir):
        return False

    print("\n" + "="*70)
    print(f"EXTRACTED FEATURES {title_suffix} - MINTÁK SZÁMA A NEURALIS HALOZATNAK")
    print(f"Mappa: {os.path.basename(processed_dir)}")
    print("="*70)
    print(f"{'Osztály':<15} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Összesen':<7}")
    print("-" * 70)

    # Load labels
    labels = {}
    for subset in ['train', 'val', 'test']:
        label_file = os.path.join(processed_dir, f"y_labels_{subset}.npy")
        if os.path.exists(label_file):
            labels[subset] = np.load(label_file)
        else:
            labels[subset] = np.array([])

    total_train = 0
    total_val = 0
    total_test = 0

    for idx, cls in enumerate(CLASSES):
        train_count = np.sum(labels['train'] == idx) if len(labels['train']) > 0 else 0
        val_count = np.sum(labels['val'] == idx) if len(labels['val']) > 0 else 0
        test_count = np.sum(labels['test'] == idx) if len(labels['test']) > 0 else 0
        
        total_cls = train_count + val_count + test_count
        
        total_train += train_count
        total_val += val_count
        total_test += test_count
        
        print(f"{cls:<15} | {train_count:<7} | {val_count:<7} | {test_count:<7} | {total_cls:<7}")

    print("-" * 70)
    total_all = total_train + total_val + total_test
    print(f"{'ÖSSZESEN':<15} | {total_train:<7} | {total_val:<7} | {total_test:<7} | {total_all:<7}")
    
    # Feature shapes check
    print("-" * 70)
    print("Feature tömbök méretei (Log-Mel):")
    for subset in ['train', 'val', 'test']:
        feat_file = os.path.join(processed_dir, f"X_log_mel_{subset}.npy")
        if os.path.exists(feat_file):
            # Load metadata/shape only using mmap to be fast and memory efficient
            shape = np.load(feat_file, mmap_mode='r').shape
            print(f"  - {subset:<5} : {shape}")
        else:
            print(f"  - {subset:<5} : Nincs adat")
            
    print("="*70 + "\n")
    return True

def main():
    printed_any = False
    if print_feature_stats(PROCESSED_DIR_CLEAN, "TISZTA (CLEAN)"):
        printed_any = True
    if print_feature_stats(PROCESSED_DIR_MIC, "MIKROFONOS (MIC)"):
        printed_any = True
        
    if not printed_any:
        print(f"Hiba: Egyik feldolgozott adat mappa sem található:\n- {PROCESSED_DIR_CLEAN}\n- {PROCESSED_DIR_MIC}")

if __name__ == "__main__":
    main()
