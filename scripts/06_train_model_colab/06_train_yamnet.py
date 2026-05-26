import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATASET_TYPE = 'mic'  # YAMNet requires 'mic' (raw audio only extracted for mic dataset)
DATA_PATH = os.path.join(PROJECT_ROOT, f'processed_data_{DATASET_TYPE}')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'models_{DATASET_TYPE}')

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
MODEL_NAME = 'yamnet_transfer'


# --- YAMNet ---
print("Loading YAMNet from TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet loaded.")


def extract_yamnet_embeddings(raw_audio_array):
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


def plot_results(y_test, y_pred, history, model_name, save_path):
    report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'best_{model_name}_confusion_matrix.png'), dpi=300)
    plt.show(); plt.close()

    f1_scores = [report_dict[cls]['f1-score'] for cls in CLASSES]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=CLASSES, y=f1_scores, hue=CLASSES, palette='viridis', legend=False)
    plt.title(f'F1-Scores - {model_name}')
    plt.ylabel('F1-Score'); plt.ylim(0, 1.1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'best_{model_name}_f1_scores.png'), dpi=300)
    plt.show(); plt.close()

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
    plt.savefig(os.path.join(save_path, f'best_{model_name}_acc_loss.png'), dpi=300)
    plt.show(); plt.close()

    return report_dict


# --- Load & Extract ---
print(f"\nLoading raw audio from {DATA_PATH}...")
X_train_raw = np.load(os.path.join(DATA_PATH, 'X_raw_train.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_labels_train.npy'))
X_val_raw = np.load(os.path.join(DATA_PATH, 'X_raw_val.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_labels_val.npy'))
X_test_raw = np.load(os.path.join(DATA_PATH, 'X_raw_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_labels_test.npy'))

print("Raw audio shapes:")
print(f"  train: {X_train_raw.shape}")
print(f"  val  : {X_val_raw.shape}")
print(f"  test : {X_test_raw.shape}\n")

print("\nExtracting embeddings (Train)...")
X_train = extract_yamnet_embeddings(X_train_raw)
print("Extracting embeddings (Val)...")
X_val = extract_yamnet_embeddings(X_val_raw)
print("Extracting embeddings (Test)...")
X_test = extract_yamnet_embeddings(X_test_raw)

print("Embedding shapes:")
print(f"  train: {X_train.shape}")
print(f"  val  : {X_val.shape}")
print(f"  test : {X_test.shape}\n")

np.save(os.path.join(DATA_PATH, 'X_yamnet_emb_train.npy'), X_train)
np.save(os.path.join(DATA_PATH, 'X_yamnet_emb_val.npy'), X_val)
np.save(os.path.join(DATA_PATH, 'X_yamnet_emb_test.npy'), X_test)
print("Embeddings saved.")

# --- Train ---
num_classes = len(CLASSES)
LABEL_SMOOTHING = 0.1
y_train_smooth = tf.one_hot(y_train, num_classes) * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
y_val_onehot = tf.one_hot(y_val, num_classes)

model = build_classifier(embedding_dim=1024, num_classes=num_classes)
model.summary()

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, f'best_{MODEL_NAME}_model.keras'),
        monitor='val_accuracy', save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("\nTraining...")
history = model.fit(
    X_train, y_train_smooth,
    validation_data=(X_val, y_val_onehot),
    epochs=200, batch_size=64,
    callbacks=callbacks_list
)

# --- Evaluate ---
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report (YAMNet Transfer Learning):")
print(classification_report(y_test, y_pred, target_names=CLASSES))

report_dict = plot_results(y_test, y_pred, history, MODEL_NAME, MODEL_SAVE_PATH)

micro_f1 = report_dict['accuracy']
macro_f1 = report_dict['macro avg']['f1-score']

print(f"YAMNet Transfer Learning - Accuracy: {micro_f1*100:.1f}%")
print(f"YAMNet Transfer Learning - Macro F1: {macro_f1*100:.1f}%")
