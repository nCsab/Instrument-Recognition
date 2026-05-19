import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATA_PATH = os.path.join(PROJECT_ROOT, 'processed_data')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models')

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
FEATURE_TYPE = 'log_mel'


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


print(f"Loading data ({FEATURE_TYPE})...")
X = np.load(os.path.join(DATA_PATH, f'X_{FEATURE_TYPE}_full.npy'))
if X.ndim == 3: X = X[..., np.newaxis]
y_file = f'y_{FEATURE_TYPE}_labels.npy'
if not os.path.exists(os.path.join(DATA_PATH, y_file)): y_file = 'y_labels_full.npy'
y = np.load(os.path.join(DATA_PATH, y_file))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = build_standard_2d_cnn_model((X.shape[1], X.shape[2], X.shape[3]), len(CLASSES))
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    callbacks.ModelCheckpoint(os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_model.keras'), monitor='val_accuracy', save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

print("\nTraining...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=callbacks_list)

print("\n===== EVALUATION =====")
y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred, target_names=CLASSES))
