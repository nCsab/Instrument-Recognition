import os
import numpy as np

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]

def print_stats(p_dir, title=""):
    if not os.path.exists(p_dir): return
    print(f"\n{title}\n{'CLASS':<15} | {'TRAIN':<7} | {'VAL':<7} | {'TEST':<7} | {'TOTAL':<7}")
    
    L = {s: (np.load(f) if os.path.exists(f := os.path.join(p_dir, f"y_labels_{s}.npy")) else []) for s in ['train', 'val', 'test']}
    t = [0, 0, 0]
    
    for i, cls in enumerate(CLASSES):
        c = [int(np.sum(L[s] == i)) if len(L[s]) else 0 for s in ['train', 'val', 'test']]
        t[0]+=c[0]; t[1]+=c[1]; t[2]+=c[2]
        print(f"{cls:<15} | {c[0]:<7} | {c[1]:<7} | {c[2]:<7} | {sum(c):<7}")
    print(f"{'TOTAL':<15} | {t[0]:<7} | {t[1]:<7} | {t[2]:<7} | {sum(t):<7}\n")

    print("Feature shapes (Log-Mel):")
    for s in ['train', 'val', 'test']:
        p = os.path.join(p_dir, f"X_log_mel_{s}.npy")
        print(f"  {s:<5}: {np.load(p, mmap_mode='r').shape if os.path.exists(p) else 'N/A'}")

if __name__ == "__main__":
    b = "/Users/csabanagy/Desktop/project"
    print_stats(os.path.join(b, "processed_data_clean"), "EXTRACTED FEATURES: CLEAN (No Aug)")
    print_stats(os.path.join(b, "processed_data_augmented"), "EXTRACTED FEATURES: AUGMENTED")
    print_stats(os.path.join(b, "processed_data_mic"), "EXTRACTED FEATURES: MIC")
