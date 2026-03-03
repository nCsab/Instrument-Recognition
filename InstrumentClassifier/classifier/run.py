#!/usr/bin/env python3

import argparse

from classifier import config
from classifier.pipeline import Pipeline


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
    print("  MONOFONIKUS HANGSZEROSZTÁLYOZÓ — NSynth Pipeline")
    print("=" * 60)
    print(f"  Hangszercsaládok: {', '.join(config.CLASS_NAMES)}")
    print(f"  Sample rate: {config.SAMPLE_RATE} Hz")
    print(f"  FFT size: {config.FFT_SIZE}")
    if args.max_per_class:
        print(f"  Max per class: {args.max_per_class}")
    print("=" * 60)

    pipeline = Pipeline(max_per_class=args.max_per_class)
    results = pipeline.run()

    print(f"\n  Végeredmény: {results['accuracy']:.1%} pontosság")


if __name__ == "__main__":
    main()
