import os
from collections import Counter

ROOT_DIR = "/Volumes/Kingston XS1000 Media/project/dataset"
DATASETS = [
    "ESC-50-master",
    "IRMAS-TrainingData",
    "IRMAS-TestingData-Part2",
    "IRMAS-TestingData-Part3",
    "TinySOL",
    "good-sounds",
    "nsynth-train",
    "Medley-solos-DB"
]

AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.m4a', '.ogg')
WRAPPERS = ['sound_files', 'audio', 'IRTestingData-Part2', 'IRTestingData-Part3', 'IRTestingData-Part1']


def analyze_dataset(dataset_name):
    dataset_path = os.path.join(ROOT_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} does not exist.")
        return

    audio_counts = Counter()
    nsynth_counts = {}
    total_audio_files = 0
    total_files = 0

    is_irmas_test = "IRMAS-TestingData-Part" in dataset_name
    is_medley = "Medley-solos-DB" in dataset_name
    is_nsynth = "nsynth" in dataset_name

    for root, dirs, files in os.walk(dataset_path):
        total_files += len(files)

        audio_files = [f for f in files if f.lower().endswith(AUDIO_EXTENSIONS)]
        if not audio_files:
            continue

        total_audio_files += len(audio_files)

        if is_irmas_test:
            txt_files = [f for f in files if f.lower().endswith('.txt') and f.lower() not in ["readme.txt", "dataset_summary.txt"]]
            for tf in txt_files:
                try:
                    with open(os.path.join(root, tf), 'r') as f:
                        labels = [line.strip() for line in f if line.strip()]
                        for label in labels:
                            audio_counts[label] += 1
                except Exception as e:
                    print(f"Error reading {tf}: {e}")
            continue

        if is_nsynth:
            for af in audio_files:
                parts = af.split('_')
                if len(parts) >= 2:
                    instrument = parts[0]
                    source = parts[1]
                    if instrument not in nsynth_counts:
                        nsynth_counts[instrument] = Counter()
                    nsynth_counts[instrument][source] += 1
                else:
                    audio_counts["unknown"] += 1
            continue

        if is_medley and root == dataset_path:
            csv_file = os.path.join(dataset_path, "Medley-solos-DB_metadata.csv")
            if os.path.exists(csv_file):
                import csv
                try:
                    with open(csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cat = row.get('instrument', 'unknown')
                            audio_counts[cat] += 1
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
            continue
        elif is_medley:
            continue

        rel_path = os.path.relpath(root, dataset_path)
        parts = rel_path.split(os.sep)

        if rel_path == ".":
            category = "root"
        else:
            category_parts = [p for p in parts if p not in WRAPPERS]
            category = category_parts[0] if category_parts else parts[-1]

        audio_counts[category] += len(audio_files)

    summary_file = os.path.join(dataset_path, "dataset_summary.md")
    with open(summary_file, "w") as f:
        f.write(f"# Dataset Summary: {dataset_name}\n\n")
        f.write(f"- **Total Files (All types):** {total_files}\n")
        f.write(f"- **Total Audio Files:** {total_audio_files}\n\n")
        f.write("## Category Breakdown\n\n")

        if is_irmas_test:
            f.write("> [!NOTE]\n")
            f.write("> For IRMAS Testing Data, categories are extracted from `.txt` annotation files.\n\n")
        elif is_medley:
            f.write("> [!NOTE]\n")
            f.write("> For Medley-solos-DB, categories are extracted from `Medley-solos-DB_metadata.csv`.\n\n")
        elif is_nsynth:
            f.write("> [!NOTE]\n")
            f.write("> For NSynth, categories and sources are extracted from filenames.\n\n")

        if is_nsynth:
            for instrument in sorted(nsynth_counts.keys()):
                sources = nsynth_counts[instrument]
                total_for_inst = sum(sources.values())
                f.write(f"### {instrument} (Total: {total_for_inst})\n")
                for src, count in sorted(sources.items()):
                    f.write(f"- {src}: {count}\n")
                f.write("\n")
        else:
            f.write("| Category | Count |\n")
            f.write("| :--- | :--- |\n")
            for cat, count in sorted(audio_counts.items()):
                f.write(f"| {cat} | {count} |\n")

    print(f"Generated summary for {dataset_name}")


if __name__ == "__main__":
    for ds in DATASETS:
        analyze_dataset(ds)
