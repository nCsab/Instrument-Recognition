from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

FFT_SIZE = 1024
HOP_SIZE = 512
WINDOW = "hann"

N_MFCC = 13
N_MELS = 40

INSTRUMENT_FAMILIES = {
    "guitar":   0,
    "keyboard": 1,
    "string":   2,
    "brass":    3,
    "reed":     4,
    "mallet":   5,
}

NUM_CLASSES = len(INSTRUMENT_FAMILIES)
CLASS_NAMES = list(INSTRUMENT_FAMILIES.keys())

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NSYNTH_DIR = DATA_DIR / "nsynth"
REPORTS_DIR = DATA_DIR / "reports"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
