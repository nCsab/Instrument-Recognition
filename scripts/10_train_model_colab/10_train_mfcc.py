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
DATASET_TYPE = 'mic'  # 'clean' vagy 'mic'
DATA_PATH = os.path.join(PROJECT_ROOT, f'processed_data_{DATASET_TYPE}')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'models_{DATASET_TYPE}')

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
FEATURE_TYPE = 'mfcc'


def build_mfcc_2d_cnn_model(input_shape, num_classes):
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


def load_subset(subset, feature_type):
    x_path = os.path.join(DATA_PATH, f'X_{feature_type}_{subset}.npy')
    y_path = os.path.join(DATA_PATH, f'y_labels_{subset}.npy')
    X_sub = np.load(x_path)
    if X_sub.ndim == 3: X_sub = X_sub[..., np.newaxis]
    y_sub = np.load(y_path)
    return X_sub, y_sub

print(f"Loading pre-split data ({FEATURE_TYPE})...")
X_train, y_train = load_subset('train', FEATURE_TYPE)
X_val, y_val = load_subset('val', FEATURE_TYPE)
X_test, y_test = load_subset('test', FEATURE_TYPE)

model = build_mfcc_2d_cnn_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]), len(CLASSES))
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    callbacks.ModelCheckpoint(os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_model.keras'), monitor='val_accuracy', save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

print("\nTraining...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=callbacks_list)
y_pred = np.argmax(model.predict(X_test), axis=1)

# Print Text Report
report_text = classification_report(y_test, y_pred, target_names=CLASSES)
print("\nClassification Report:")
print(report_text)

# Get Dict for Plotting
report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)

# Explicitly print Macro and Micro F1
micro_f1 = report_dict['accuracy']
macro_f1 = report_dict['macro avg']['f1-score']
print(f"Extracted Metrics:")
print(f" - Micro F1 (Overall Accuracy): {micro_f1:.4f}")
print(f" - Macro F1 (Unweighted Average): {macro_f1:.4f}\n")

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 1. Save and Show Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'Confusion Matrix - {FEATURE_TYPE} 2D-CNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plot_path = os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_confusion_matrix.png')
plt.savefig(plot_path, dpi=300)
plt.show()
plt.close()
print(f"\nConfusion matrix heatmap saved successfully to {plot_path}")

# 2. Save and Show F1-Score Bar Chart
f1_scores = [report_dict[cls]['f1-score'] for cls in CLASSES]
plt.figure(figsize=(10, 6))
sns.barplot(x=CLASSES, y=f1_scores, palette='viridis')
plt.title(f'F1-Scores per Class - {FEATURE_TYPE}')
plt.ylabel('F1-Score')
plt.ylim(0, 1.1)
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
f1_plot_path = os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_f1_scores.png')
plt.savefig(f1_plot_path, dpi=300)
plt.show()
plt.close()
print(f"F1-score bar chart saved successfully to {f1_plot_path}")

# Save Accuracy and Loss plots
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f'Model Accuracy - {FEATURE_TYPE}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss - {FEATURE_TYPE}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
acc_loss_path = os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_acc_loss.png')
plt.savefig(acc_loss_path, dpi=300)
plt.show()
plt.close()
print(f"Accuracy and Loss plots saved successfully to {acc_loss_path}")
