import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATASET_TYPE = 'mic'
DATA_PATH = os.path.join(PROJECT_ROOT, f'processed_data_{DATASET_TYPE}')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'models_{DATASET_TYPE}')

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
FEATURE_TYPE = 'stft'


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def load_subset(subset, feature_type):
    X = np.load(os.path.join(DATA_PATH, f'X_{feature_type}_{subset}.npy'))
    if X.ndim == 3: X = X[..., np.newaxis]
    y = np.load(os.path.join(DATA_PATH, f'y_labels_{subset}.npy'))
    return X, y


def plot_results(y_test, y_pred, history, feature_type, save_path):
    report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - {feature_type}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'best_{feature_type}_2dcnn_confusion_matrix.png'), dpi=300)
    plt.show(); plt.close()

    f1_scores = [report_dict[cls]['f1-score'] for cls in CLASSES]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=CLASSES, y=f1_scores, palette='viridis')
    plt.title(f'F1-Scores - {feature_type}')
    plt.ylabel('F1-Score'); plt.ylim(0, 1.1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'best_{feature_type}_2dcnn_f1_scores.png'), dpi=300)
    plt.show(); plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'Accuracy - {feature_type}'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'Loss - {feature_type}'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'best_{feature_type}_2dcnn_acc_loss.png'), dpi=300)
    plt.show(); plt.close()

    return report_dict


# --- Main ---
print(f"Loading data ({FEATURE_TYPE})...")
X_train, y_train = load_subset('train', FEATURE_TYPE)
X_val, y_val = load_subset('val', FEATURE_TYPE)
X_test, y_test = load_subset('test', FEATURE_TYPE)

model = build_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]), len(CLASSES))
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    callbacks.ModelCheckpoint(os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_model.keras'),
                              monitor='val_accuracy', save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

print("\nTraining...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=100, batch_size=32, callbacks=callbacks_list)

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

report_dict = plot_results(y_test, y_pred, history, FEATURE_TYPE, MODEL_SAVE_PATH)
print(f"Accuracy: {report_dict['accuracy']*100:.1f}%")
print(f"Macro F1: {report_dict['macro avg']['f1-score']*100:.1f}%")
