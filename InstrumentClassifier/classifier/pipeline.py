import numpy as np
from pathlib import Path

from classifier import config
from classifier.features.mfcc_features import MFCCFeatureExtractor
from classifier.models.random_forest import RandomForestStrategy
from classifier.dataset.loader import NsynthLoader
from classifier.evaluation.metrics import evaluate_model


class Pipeline:

    def __init__(self, data_dir=None, max_per_class=None):
        self.feature_extractor = MFCCFeatureExtractor()
        self.classifier = RandomForestStrategy()
        self.loader = NsynthLoader(data_dir=data_dir, max_per_class=max_per_class)

    def run(self):
        print(f"\n{'='*60}")
        print(f"  Pipeline: MFCC + RandomForest")
        print(f"{'='*60}")

        print("\n[1/4] Loading dataset...")
        splits = self.loader.load_and_split()

        print("\n[2/4] Extracting features...")
        X_train, y_train = self._extract_features_batch(
            splits["train"][0], splits["train"][1], "train"
        )
        X_val, y_val = self._extract_features_batch(
            splits["val"][0], splits["val"][1], "val"
        )
        X_test, y_test = self._extract_features_batch(
            splits["test"][0], splits["test"][1], "test"
        )

        print(f"  Feature dimension: {X_train.shape[1]}")
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        print("\n[3/4] Training classifier...")
        self.classifier.train(X_train, y_train)
        print("  Training complete.")

        val_pred = self.classifier.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        print(f"  Validation accuracy: {val_acc:.4f}")

        print("\n[4/4] Evaluating on test set...")
        y_pred = self.classifier.predict(X_test)
        y_proba = self.classifier.predict_proba(X_test)

        results = evaluate_model(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name="MFCC_RandomForest",
        )

        return results

    def predict(self, audio, sr=None):
        sr = sr or config.SAMPLE_RATE
        features = self.feature_extractor.extract(audio, sr)
        features_2d = features.reshape(1, -1)

        pred = self.classifier.predict(features_2d)[0]
        proba = self.classifier.predict_proba(features_2d)[0]
        confidence = float(proba[pred])
        instrument = config.CLASS_NAMES[pred]

        return instrument, confidence

    def _extract_features_batch(self, file_paths, labels, split_name):
        import librosa

        sr = config.SAMPLE_RATE
        duration = config.DURATION

        features_list = []
        valid_labels = []

        for i, fp in enumerate(file_paths):
            try:
                audio, _ = librosa.load(fp, sr=sr, duration=duration)
                feat = self.feature_extractor.extract(audio, sr)
                features_list.append(feat)
                valid_labels.append(labels[i])
            except Exception as e:
                print(f"  Warning: skipping {fp}: {e}")

            if (i + 1) % 200 == 0:
                print(f"  [{split_name}] Extracted {i + 1}/{len(file_paths)}...")

        print(f"  [{split_name}] Done: {len(features_list)} features extracted.")
        return np.array(features_list), np.array(valid_labels)
