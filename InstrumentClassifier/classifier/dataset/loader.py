import json
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split

from classifier import config


class NsynthLoader:

    def __init__(self, data_dir=None, max_per_class=None):
        self.data_dir = Path(data_dir) if data_dir else config.NSYNTH_DIR
        self.max_per_class = max_per_class

    def load_split(self, split="train"):
        split_dir = self.data_dir / f"nsynth-{split}"
        audio_dir = split_dir / "audio"
        json_path = split_dir / "examples.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Metadata not found: {json_path}")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        file_paths = []
        labels = []
        family_counts = {name: 0 for name in config.INSTRUMENT_FAMILIES}

        for note_id, info in metadata.items():
            family = info.get("instrument_family_str", "")
            source = info.get("instrument_source_str", "")

            if family not in config.INSTRUMENT_FAMILIES:
                continue
            if source != "acoustic":
                continue
            if self.max_per_class and family_counts[family] >= self.max_per_class:
                continue

            wav_path = audio_dir / f"{note_id}.wav"
            if wav_path.exists():
                file_paths.append(wav_path)
                labels.append(config.INSTRUMENT_FAMILIES[family])
                family_counts[family] += 1

        return file_paths, np.array(labels)

    def load_and_split(self, max_per_class=None):
        mpc = max_per_class or self.max_per_class

        available_splits = []
        for split_name in ["train", "valid", "test"]:
            split_dir = self.data_dir / f"nsynth-{split_name}"
            if (split_dir / "examples.json").exists():
                available_splits.append(split_name)

        if not available_splits:
            raise FileNotFoundError(
                f"No NSynth splits found in {self.data_dir}. "
                "Run download_nsynth.py first."
            )

        if "train" in available_splits:
            train_paths, train_labels = self.load_split("train")
            val_paths, val_labels = self.load_split("valid")
            test_paths, test_labels = self.load_split("test")
        else:
            all_paths = []
            all_labels = []

            for split_name in available_splits:
                paths, labels = self.load_split(split_name)
                all_paths.extend(paths)
                all_labels.extend(labels)

            all_labels = np.array(all_labels)

            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                all_paths, all_labels,
                test_size=0.30,
                stratify=all_labels,
                random_state=config.RANDOM_STATE,
            )
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels,
                test_size=0.50,
                stratify=temp_labels,
                random_state=config.RANDOM_STATE,
            )

        if mpc:
            train_paths, train_labels = self._limit_per_class(train_paths, train_labels, mpc)

        return {
            "train": (train_paths, train_labels),
            "val": (val_paths, val_labels),
            "test": (test_paths, test_labels),
        }

    def _limit_per_class(self, paths, labels, max_per_class):
        counts = {}
        filtered_paths = []
        filtered_labels = []

        for p, l in zip(paths, labels):
            counts[l] = counts.get(l, 0) + 1
            if counts[l] <= max_per_class:
                filtered_paths.append(p)
                filtered_labels.append(l)

        return filtered_paths, np.array(filtered_labels)
