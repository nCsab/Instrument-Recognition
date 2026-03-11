#!/usr/bin/env python3

import sys
sys.argv = [sys.argv[0]] + sys.argv[1:]

import argparse
from classifier import config
from classifier.pipeline import Pipeline
from classifier.features.mfcc_features import MFCCFeatureExtractor
from classifier.features.fft_features import FFTFeatureExtractor
from classifier.models.random_forest import RandomForestStrategy


def main():
    parser = argparse.ArgumentParser(
        description="Run instrument classification pipeline on NSynth data"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Max samples per instrument family (for faster testing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MONOFONIKUS HANGSZEROSZTÁLYOZÓ — ÖSSZEHASONLÍTÁS")
    print("=" * 60)
    print(f"  Hangszercsaládok: {', '.join(config.CLASS_NAMES)}")
    print(f"  Sample rate: {config.SAMPLE_RATE} Hz")
    if args.max_per_class:
        print(f"  Max per class: {args.max_per_class}")
    print("=" * 60)

    extractors = {
        "MFCC": MFCCFeatureExtractor(),
        "FFT":  FFTFeatureExtractor(),
    }

    results_summary = []

    for name, extractor in extractors.items():
        print(f"\n{'='*60}")
        print(f"  Pipeline: {name} + RandomForest")
        print(f"{'='*60}")

        pipeline = Pipeline(
            feature_extractor=extractor,
            classifier=RandomForestStrategy(),
            max_per_class=args.max_per_class,
        )
        results = pipeline.run()

        results_summary.append({
            "feature": name,
            "accuracy": results["accuracy"],
            "dim": extractor.get_feature_dim(),
        })

    print(f"\n{'='*60}")
    print(f"  ÖSSZEHASONLÍTÁS")
    print(f"{'='*60}")
    print(f"  {'Feature':<12} {'Dim':>5} {'Pontosság':>12}")
    print(f"  {'-'*32}")
    for r in sorted(results_summary, key=lambda x: x["accuracy"], reverse=True):
        print(f"  {r['feature']:<12} {r['dim']:>5} {r['accuracy']:>11.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
