"""
GOOGLE COLAB TANÍTÓ TEMPLATE (MULTI-MODEL)
-------------------------------------------
Ezt a teljes fájlt bemásolhatod egy Google Colab cellába.
Tartalmazza a Standard (Log-Mel/STFT) és a Speciális (MFCC) 2D CNN-t is.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURÁCIÓ ---
CLASSES = ["guitar", "piano", "other", "noise"]

def build_standard_2d_cnn_model(input_shape, num_classes):
    """
    Standard 2D CNN - Log-Mel és STFT spektrogramokhoz optimalizálva.
    Nagyobb pooling-ot használ a frekvenciák összevonásához.
    """
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

def build_mfcc_2d_cnn_model(input_shape, num_classes):
    """
    Speciális 2D CNN - MFCC-hez optimalizálva. 
    Kisebb pooling (több információ), kisebb Dropout és sávonkénti normalizálás (CMVN) mellé szánva.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)), 
        layers.Dropout(0.2),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1, 2)), 
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_colab_training(data_path, model_save_path, feature_type='mfcc'):
    """
    Teljes tanítási folyamat.
    feature_type: 'mfcc', 'log_mel' vagy 'stft'
    """
    print(f"Tanítás indítása: {feature_type} jellemzőkkel...")
    
    # 1. Adatok betöltése
    X_file = f'X_{feature_type}_full.npy'
    y_file = f'y_{feature_type}_labels.npy' if feature_type == 'mfcc' else 'y_labels_full.npy'
    
    X = np.load(os.path.join(data_path, X_file))
    y = np.load(os.path.join(data_path, y_file))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Modell kiválasztása
    input_shape = (X.shape[1], X.shape[2], 1)
    if feature_type == 'mfcc':
        model = build_mfcc_2d_cnn_model(input_shape, len(CLASSES))
    else:
        model = build_standard_2d_cnn_model(input_shape, len(CLASSES))

    if not os.path.exists(model_save_path): os.makedirs(model_save_path)

    # 3. Tanítás
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(os.path.join(model_save_path, f'best_{feature_type}_model.keras'), save_best_only=True)

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=50, batch_size=32, callbacks=[early_stop, checkpoint]
    )

    # 4. Kiértékelés és Vizualizáció
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {feature_type}')
    plt.show()

    print(f"\nClassification Report ({feature_type}):")
    print(classification_report(y_test, y_pred_classes, target_names=CLASSES))

# HASZNÁLAT COLAB-BAN:
# DATA_PATH = '/content/drive/MyDrive/Instrument_Recognition/processed_data'
# SAVE_PATH = '/content/drive/MyDrive/Instrument_Recognition/models'
# run_colab_training(DATA_PATH, SAVE_PATH, feature_type='mfcc')
