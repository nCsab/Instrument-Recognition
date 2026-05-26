import os

def print_stats(dataset_dir, title):
    if not os.path.exists(dataset_dir): return
    classes = sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)))
    
    print(f"\n{title}\n{'CLASS':<15} | {'TRAIN':<7} | {'VAL':<7} | {'TEST':<7} | {'TOTAL':<7}")
    t_tr, t_vl, t_ts = 0, 0, 0
    for cls in classes:
        c = [len([f for f in os.listdir(os.path.join(dataset_dir, cls, s)) if f.endswith('.wav')]) if os.path.exists(os.path.join(dataset_dir, cls, s)) else 0 for s in ['train', 'val', 'test']]
        t_tr += c[0]; t_vl += c[1]; t_ts += c[2]
        print(f"{cls:<15} | {c[0]:<7} | {c[1]:<7} | {c[2]:<7} | {sum(c):<7}")
    print(f"{'TOTAL':<15} | {t_tr:<7} | {t_vl:<7} | {t_ts:<7} | {t_tr+t_vl+t_ts:<7}\n")

if __name__ == "__main__":
    b_dir = "/Users/csabanagy/Desktop/project"
    print_stats(os.path.join(b_dir, "dataset_clean"), "CLEAN DATASET\n")
    print_stats(os.path.join(b_dir, "dataset_mic"), "MIC DATASET\n")
