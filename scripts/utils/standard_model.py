import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

# 1. Google Drive felcsatolása
drive.mount('/content/drive')

# --- ÚTVONALAK ÉS KONFIGURÁCIÓ ---
PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATA_PATH = os.path.join(PROJECT_ROOT, 'processed_data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models')
CLASSES = ["guitar", "piano", "other", "noise"]

# Válaszd ki: 'log_mel' vagy 'stft'
FEATURE_TYPE = 'log_mel' 

# 2. Adatok betöltése
print(f"Adatok betöltése ({FEATURE_TYPE})...")
X = np.load(os.path.join(DATA_PATH, f'X_{FEATURE_TYPE}_full.npy'))
# Batch kinyerésnél y_labels_full.npy, egyéninél y_log_mel_labels.npy/y_stft_labels.npy
y_file = f'y_{FEATURE_TYPE}_labels.npy'
if not os.path.exists(os.path.join(DATA_PATH, y_file)):
    y_file = 'y_labels_full.npy'
y = np.load(os.path.join(DATA_PATH, y_file))

# Split: 80% tanító / 20% tesztelő
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Modell definíciója (Standard 2D CNN)
def build_standard_2d_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Tanítás Futtatása
input_shape = (X.shape[1], X.shape[2], 1)
model = build_standard_2d_cnn_model(input_shape, len(CLASSES))

if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_model.keras'), save_best_only=True)

print("\nTanítás indítása...")
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test),
    epochs=50, batch_size=32, callbacks=[early_stop, checkpoint]
)

# 5. --- KIÉRTÉKELÉS (Confusion Matrix) ---
print("\nÉrtékelés...")
y_pred = np.argmax(model.predict(X_test), axis=1)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {FEATURE_TYPE}')
plt.show()

# Statisztikák
print(classification_report(y_test, y_pred, target_names=CLASSES))

# Görbék
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()
