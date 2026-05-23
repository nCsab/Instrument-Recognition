"""
10_train_yamnet.py - Transfer Learning YAMNet-tel (Google Colab)

YAMNet: Google által fejlesztett, az AudioSet adatbázison (2M+ klip, 521 osztály)
előtanított hangfelismerő modell. A MobileNet v1 architektúrán alapul.

Működés:
1. Betöltjük a nyers audio szegmenseket (.npy) a Drive-ról
2. A YAMNet minden szegmensből kinyeri a 1024 dimenziós embedding vektort
3. Erre a gazdag, előtanított jellemzőtérre egy kisebb, saját osztályozó hálózatot tanítunk
4. Az eredmény összehasonlítható a nulláról tanított CNN modellünkkel

Előny: A YAMNet már "érti" a hangok szerkezetét, ezért sokkal kevesebb adatból
is jobb eredményt képes elérni, mint egy üres hálózat.
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATASET_TYPE = 'mic'  # 'clean' vagy 'mic'
DATA_PATH = os.path.join(PROJECT_ROOT, f'processed_data_{DATASET_TYPE}')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'models_{DATASET_TYPE}')

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
MODEL_NAME = 'yamnet_transfer'

# ============================================================
# YAMNet betöltése TensorFlow Hub-ról
# ============================================================
print("Loading YAMNet from TensorFlow Hub...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
print("YAMNet loaded successfully!")


# ============================================================
# Embedding kinyerés YAMNet-tel
# Minden nyers audio szegmensből egy 1024 dimenziós vektor lesz
# ============================================================
def extract_yamnet_embeddings(raw_audio_array):
    """
    Nyers audio szegmensekből YAMNet embedding vektorokat nyer ki.

    Args:
        raw_audio_array: np.array, shape (N, 16000) - N db 1 másodperces szegmens

    Returns:
        np.array, shape (N, 1024) - N db embedding vektor
    """
    embeddings_list = []
    total = len(raw_audio_array)

    for i, waveform in enumerate(raw_audio_array):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Embedding extraction: {i + 1}/{total}")

        # YAMNet-nek float32 kell, 16kHz mintavétel
        waveform_tf = tf.cast(waveform, tf.float32)

        # YAMNet kimenetei: scores (521 AudioSet osztály), embeddings (1024-dim), spectrogram
        scores, embeddings, spectrogram = yamnet_model(waveform_tf)

        # Ha több frame van (1 másodpercre általában 1), átlagoljuk őket
        mean_embedding = tf.reduce_mean(embeddings, axis=0)
        embeddings_list.append(mean_embedding.numpy())

    return np.array(embeddings_list)


# ============================================================
# Osztályozó modell a YAMNet embedding-ekre
# Egyszerű Dense hálózat, mert a YAMNet már elvégezte a
# nehéz jellemzőkinyerést
# ============================================================
def build_yamnet_classifier(embedding_dim=1024, num_classes=7):
    model = models.Sequential([
        layers.Input(shape=(embedding_dim,)),

        # 1. rejtett réteg
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # 2. rejtett réteg
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        # Kimeneti réteg
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


# ============================================================
# Adatok betöltése és YAMNet embedding kinyerés
# ============================================================
print(f"\nLoading raw audio segments from {DATA_PATH}...")

X_train_raw = np.load(os.path.join(DATA_PATH, 'X_raw_train.npy'))
y_train = np.load(os.path.join(DATA_PATH, 'y_labels_train.npy'))
X_val_raw = np.load(os.path.join(DATA_PATH, 'X_raw_val.npy'))
y_val = np.load(os.path.join(DATA_PATH, 'y_labels_val.npy'))
X_test_raw = np.load(os.path.join(DATA_PATH, 'X_raw_test.npy'))
y_test = np.load(os.path.join(DATA_PATH, 'y_labels_test.npy'))

print(f"Raw audio - Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}")

# YAMNet embedding kinyerés (ez néhány percet vehet igénybe)
print("\nExtracting YAMNet embeddings (Train)...")
X_train = extract_yamnet_embeddings(X_train_raw)
print("Extracting YAMNet embeddings (Val)...")
X_val = extract_yamnet_embeddings(X_val_raw)
print("Extracting YAMNet embeddings (Test)...")
X_test = extract_yamnet_embeddings(X_test_raw)

print(f"\nEmbeddings - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Embedding-ek mentése (hogy ne kelljen újra kinyerni, ha újratanítunk)
np.save(os.path.join(DATA_PATH, 'X_yamnet_emb_train.npy'), X_train)
np.save(os.path.join(DATA_PATH, 'X_yamnet_emb_val.npy'), X_val)
np.save(os.path.join(DATA_PATH, 'X_yamnet_emb_test.npy'), X_test)
print("Embeddings saved for future use!")

# ============================================================
# Tanítás
# ============================================================
num_classes = len(CLASSES)
LABEL_SMOOTHING = 0.1
y_train_smooth = tf.one_hot(y_train, num_classes) * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
y_val_onehot = tf.one_hot(y_val, num_classes)
y_test_onehot = tf.one_hot(y_test, num_classes)

model = build_yamnet_classifier(embedding_dim=1024, num_classes=num_classes)
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

print("\nTraining YAMNet classifier...")
history = model.fit(
    X_train, y_train_smooth,
    validation_data=(X_val, y_val_onehot),
    epochs=200,
    batch_size=64,
    callbacks=callbacks_list
)

# ============================================================
# Értékelés
# ============================================================
y_pred = np.argmax(model.predict(X_test), axis=1)

report_text = classification_report(y_test, y_pred, target_names=CLASSES)
print("\nClassification Report (YAMNet Transfer Learning):")
print(report_text)

report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)

micro_f1 = report_dict['accuracy']
macro_f1 = report_dict['macro avg']['f1-score']
print(f"Extracted Metrics:")
print(f" - Micro F1 (Overall Accuracy): {micro_f1:.4f}")
print(f" - Macro F1 (Unweighted Average): {macro_f1:.4f}\n")

# ============================================================
# Vizualizáció
# ============================================================
cm = confusion_matrix(y_test, y_pred)

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Confusion Matrix - YAMNet Transfer Learning')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plot_path = os.path.join(MODEL_SAVE_PATH, f'best_{MODEL_NAME}_confusion_matrix.png')
plt.savefig(plot_path, dpi=300)
plt.show()
plt.close()
print(f"\nConfusion matrix saved to {plot_path}")

# 2. F1-Score Bar Chart
f1_scores = [report_dict[cls]['f1-score'] for cls in CLASSES]
plt.figure(figsize=(10, 6))
sns.barplot(x=CLASSES, y=f1_scores, hue=CLASSES, palette='viridis', legend=False)
plt.title(f'F1-Scores per Class - YAMNet Transfer Learning')
plt.ylabel('F1-Score')
plt.ylim(0, 1.1)
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
f1_path = os.path.join(MODEL_SAVE_PATH, f'best_{MODEL_NAME}_f1_scores.png')
plt.savefig(f1_path, dpi=300)
plt.show()
plt.close()
print(f"F1-score chart saved to {f1_path}")

# 3. Accuracy and Loss plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Model Accuracy - YAMNet Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss - YAMNet Transfer Learning')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
acc_loss_path = os.path.join(MODEL_SAVE_PATH, f'best_{MODEL_NAME}_acc_loss.png')
plt.savefig(acc_loss_path, dpi=300)
plt.show()
plt.close()
print(f"Accuracy and Loss plots saved to {acc_loss_path}")

# ============================================================
# Összehasonlító összegzés
# ============================================================
print("\n" + "=" * 70)
print("ÖSSZEHASONLÍTÁS")
print("=" * 70)
print(f"YAMNet Transfer Learning - Test Accuracy: {micro_f1*100:.1f}%")
print(f"YAMNet Transfer Learning - Macro F1:      {macro_f1*100:.1f}%")
print(f"\n(Korábbi saját CNN eredmény: ~78% test accuracy)")
print(f"(Javulás: {(micro_f1 - 0.78)*100:+.1f} százalékpont)")
print("=" * 70)
