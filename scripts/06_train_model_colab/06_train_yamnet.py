import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import random

# YAMNet transfer learning referencia Google Colabhoz.
# A YAMNet előtanított része feature extractorként működik, erre tanul egy kis saját osztályozófej.

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATASET_TYPE = 'exp_final'  # YAMNethez raw audio kell, ezt csak az exp_final menti.
EVALUATION_SPLIT = 'val'  # Modellválasztáshoz: 'val'. Csak a végső győztesnél: 'test'.
TRAIN_MODEL = True
CHECKPOINT_TO_EVALUATE = None  # Pl.: 'exp_final_yamnet_transfer_val_20260528_101010_best_model.keras'
DATA_PATH = os.path.join(PROJECT_ROOT, 'processed_data', DATASET_TYPE)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', DATASET_TYPE)

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
MODEL_NAME = 'yamnet_transfer'


# --- YAMNet ---
print("Loading YAMNet from TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet loaded.")


def extract_yamnet_embeddings(raw_audio_array):
    # Nyers 1 másodperces hangokból 1024 dimenziós YAMNet embeddingeket készít.
    embeddings_list = []
    total = len(raw_audio_array)
    for i, waveform in enumerate(raw_audio_array):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total}")
        waveform_tf = tf.cast(waveform, tf.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform_tf)
        embeddings_list.append(tf.reduce_mean(embeddings, axis=0).numpy())
    return np.array(embeddings_list)


def build_classifier(embedding_dim=1024, num_classes=7):
    # Kis Dense osztályozófej a YAMNet embeddingek tetején a hét saját audioosztályhoz.
    model = models.Sequential([
        layers.Input(shape=(embedding_dim,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def plot_results(y_true, y_pred, history, model_name, eval_split, save_path, artifact_prefix):
    # Ugyanazokat a riportábrákat menti, mint a saját CNN tanítószkript.
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - {model_name} ({eval_split})')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{artifact_prefix}_confusion_matrix.png'), dpi=300)
    plt.show(); plt.close()

    f1_scores = [report_dict[cls]['f1-score'] for cls in CLASSES]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=CLASSES, y=f1_scores, hue=CLASSES, palette='viridis', legend=False)
    plt.title(f'F1-Scores - {model_name} ({eval_split})')
    plt.ylabel('F1-Score'); plt.ylim(0, 1.1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{artifact_prefix}_f1_scores.png'), dpi=300)
    plt.show(); plt.close()

    if history is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title(f'Accuracy - {model_name}'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Loss - {model_name}'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{artifact_prefix}_acc_loss.png'), dpi=300)
        plt.show(); plt.close()

    return report_dict


def save_evaluation_report(y_true, y_pred, report_text, report_dict, eval_split, save_path, artifact_prefix, checkpoint_path):
    # Időbélyeges riportmentés, hogy több futtatás ne írja felül egymást.
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(save_path, f'{artifact_prefix}_confusion_matrix.csv')
    txt_path = os.path.join(save_path, f'{artifact_prefix}_classification_report.txt')
    json_path = os.path.join(save_path, f'{artifact_prefix}_classification_report.json')

    np.savetxt(cm_path, cm, fmt='%d', delimiter=',')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {DATASET_TYPE}\n")
        f.write("Feature: yamnet_embedding\n")
        f.write(f"Evaluation split: {eval_split}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(report_text)
        f.write(f"\nAccuracy: {report_dict['accuracy']*100:.1f}%\n")
        f.write(f"Macro F1: {report_dict['macro avg']['f1-score']*100:.1f}%\n")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": DATASET_TYPE,
            "feature": "yamnet_embedding",
            "evaluation_split": eval_split,
            "model": MODEL_NAME,
            "checkpoint": checkpoint_path,
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    print(f"Saved report: {txt_path}")
    print(f"Saved report data: {json_path}")
    print(f"Saved confusion matrix CSV: {cm_path}")


def resolve_checkpoint_path(checkpoint_name):
    if not checkpoint_name:
        return None
    return checkpoint_name if os.path.isabs(checkpoint_name) else os.path.join(MODEL_SAVE_PATH, checkpoint_name)


def load_or_extract_embeddings(split):
    # Az embedding kinyerése lassabb művelet, ezért split szerint cache-eljük .npy fájlba.
    emb_path = os.path.join(DATA_PATH, f'X_yamnet_emb_{split}.npy')
    if os.path.exists(emb_path):
        print(f"Loading cached YAMNet embeddings ({split})...")
        return np.load(emb_path)

    raw_path = os.path.join(DATA_PATH, f'X_raw_{split}.npy')
    print(f"Loading raw audio ({split})...")
    raw_audio = np.load(raw_path)
    print(f"Extracting embeddings ({split})...")
    embeddings = extract_yamnet_embeddings(raw_audio)
    np.save(emb_path, embeddings)
    return embeddings


# --- Load & Extract ---
if EVALUATION_SPLIT not in ['val', 'test']:
    raise ValueError("EVALUATION_SPLIT must be 'val' or 'test'")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
artifact_prefix = f'{DATASET_TYPE}_{MODEL_NAME}_{EVALUATION_SPLIT}_{run_id}'
checkpoint_path = resolve_checkpoint_path(CHECKPOINT_TO_EVALUATE)

print(f"\nLoading YAMNet data from {DATA_PATH}...")
y_train = np.load(os.path.join(DATA_PATH, 'y_labels_train.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_labels_val.npy'))

if EVALUATION_SPLIT == 'test':
    y_eval = np.load(os.path.join(DATA_PATH, 'y_labels_test.npy'))
else:
    y_eval = y_val

if TRAIN_MODEL:
    X_train = load_or_extract_embeddings('train')
    X_val = load_or_extract_embeddings('val')
elif checkpoint_path is None:
    raise ValueError("Set CHECKPOINT_TO_EVALUATE when TRAIN_MODEL is False.")

if EVALUATION_SPLIT == 'test':
    X_eval = load_or_extract_embeddings('test')
elif not TRAIN_MODEL:
    X_eval = load_or_extract_embeddings('val')
else:
    X_eval = X_val

# --- Train ---
num_classes = len(CLASSES)
LABEL_SMOOTHING = 0.1
if TRAIN_MODEL:
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, f'{artifact_prefix}_best_model.keras')
    y_train_smooth = tf.one_hot(y_train, num_classes) * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
    y_val_onehot = tf.one_hot(y_val, num_classes)

    print("Embedding shapes:")
    print(f"  train: {X_train.shape}")
    print(f"  val  : {X_val.shape}")
    if EVALUATION_SPLIT == 'test':
        print(f"  test : {X_eval.shape}\n")
    else:
        print("  test : not loaded during model selection\n")

    model = build_classifier(embedding_dim=1024, num_classes=num_classes)
    model.summary()

    # A validációs loss dönti el a legjobb checkpointot; a test itt sem vesz részt modellválasztásban.
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss', mode='min', save_best_only=True
        ),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    print(f"\nTraining... checkpoint: {checkpoint_path}")
    history = model.fit(
        X_train, y_train_smooth,
        validation_data=(X_val, y_val_onehot),
        epochs=200, batch_size=64,
        callbacks=callbacks_list
    )
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
else:
    print(f"\nLoading checkpoint without training: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    history = None

# --- Evaluate ---
y_pred = np.argmax(model.predict(X_eval), axis=1)

report_text = classification_report(y_eval, y_pred, target_names=CLASSES)

print(f"\nClassification Report (YAMNet Transfer Learning, {EVALUATION_SPLIT}):")
print(report_text)

report_dict = plot_results(y_eval, y_pred, history, MODEL_NAME, EVALUATION_SPLIT, MODEL_SAVE_PATH, artifact_prefix)
save_evaluation_report(y_eval, y_pred, report_text, report_dict, EVALUATION_SPLIT, MODEL_SAVE_PATH, artifact_prefix, checkpoint_path)

micro_f1 = report_dict['accuracy']
macro_f1 = report_dict['macro avg']['f1-score']

print(f"YAMNet Transfer Learning - Accuracy: {micro_f1*100:.1f}%")
print(f"YAMNet Transfer Learning - Macro F1: {macro_f1*100:.1f}%")
