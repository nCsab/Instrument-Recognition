# Instrument Recognition

This repository contains the source code and thesis documentation for a BSc
project on real-time musical instrument family recognition from audio
recordings.

The main deployed pipeline classifies one-second microphone windows into seven
classes: `guitar`, `piano`, `vocal`, `string`, `reed`, `brass`, and `noise`.
It uses normalized Log-Mel spectrograms and a compact custom 2D CNN. A
YAMNet-based transfer-learning model is included as an offline comparison, not
as the main real-time application.

## Pipeline

Run the preparation scripts from the project root in this order:

```bash
python3 scripts/01_prepare_clean_dataset.py
python3 scripts/02_acquire_mic_data.py
python3 scripts/03a_merge_mic_to_train.py
python3 scripts/03b_check_dataset_counts.py
python3 scripts/04a_extract_features.py
python3 scripts/04b_check_extracted_counts.py
```

The microphone acquisition script is interactive. It prepares playback files,
records the replayed audio, and slices the recordings for the `train`, `val`,
and `test` splits.

Model training and evaluation run in Google Colab:

- `scripts/06_train_model_colab/06_train_custom_cnn.py`
- `scripts/06_train_model_colab/06_train_yamnet.py`

## Real-Time Recognition

Install the local dependencies, then start the main application:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/05a_realtime_recognition.py
```

The real-time script loads the selected `exp_final` Log-Mel checkpoint and
continuously predicts from the laptop microphone.

## Local Data

Audio datasets, extracted NumPy arrays, generated previews, and trained model
checkpoints are intentionally excluded from Git because they are large
generated or locally collected artifacts. Their expected locations are:

- `owndataset/`
- `dataset_clean/`
- `experiment_datasets/`
- `processed_data/`
- `models/`
- `augmented_previews/`

The thesis source is available in `thesis/`.
