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
DATASET_TYPE = 'mic'  # 'clean' vagy 'mic'
DATA_PATH = os.path.join(PROJECT_ROOT, f'processed_data_{DATASET_TYPE}')
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, f'models_{DATASET_TYPE}')

CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]
FEATURE_TYPE = 'log_mel'


# ============================================================
# SpecAugment réteg: véletlenszerű frekvencia- és idő-maszkolás
# Csak a tanítás során aktív, teszteléskor kikapcsol.
# Ez megakadályozza, hogy a hálózat "bemagoljon" konkrét mintázatokat.
# ============================================================
class SpecAugment(layers.Layer):
    def __init__(self, freq_mask_param=15, time_mask_param=8, num_masks=2, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def call(self, inputs, training=None):
        if not training:
            return inputs

        augmented = inputs
        freq_dim = tf.shape(inputs)[1]
        time_dim = tf.shape(inputs)[2]

        for _ in range(self.num_masks):
            # Frekvencia maszkolás: véletlenszerű frekvenciasávok kinullázása
            f = tf.random.uniform([], 1, self.freq_mask_param, dtype=tf.int32)
            f = tf.minimum(f, freq_dim)
            f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)
            indices = tf.range(freq_dim)
            freq_mask = tf.cast(tf.logical_or(indices < f0, indices >= f0 + f), tf.float32)
            freq_mask = tf.reshape(freq_mask, [1, -1, 1, 1])
            augmented = augmented * freq_mask

            # Idő maszkolás: véletlenszerű időlépések kinullázása
            t = tf.random.uniform([], 1, self.time_mask_param, dtype=tf.int32)
            t = tf.minimum(t, time_dim)
            t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)
            indices_t = tf.range(time_dim)
            time_mask = tf.cast(tf.logical_or(indices_t < t0, indices_t >= t0 + t), tf.float32)
            time_mask = tf.reshape(time_mask, [1, 1, -1, 1])
            augmented = augmented * time_mask

        return augmented

    def get_config(self):
        config = super().get_config()
        config.update({
            'freq_mask_param': self.freq_mask_param,
            'time_mask_param': self.time_mask_param,
            'num_masks': self.num_masks,
        })
        return config


# ============================================================
# Javított modell architektúra:
# - Dupla konvolúciós blokkok (mélyebb jellemzőkinyerés)
# - GlobalAveragePooling2D a Flatten helyett (kevesebb paraméter, kevesebb overfitting)
# - Erősebb regularizáció (Dropout 0.3/0.5 + weight decay az AdamW-ben)
# - SpecAugment a bemenet után (csak tanítás közben aktív)
# - Label smoothing (0.1) a túlzott magabiztosság ellen
# ============================================================
def build_improved_2d_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # SpecAugment (csak tanítás közben aktív)
        SpecAugment(freq_mask_param=15, time_mask_param=8, num_masks=2),

        # 1. konvolúciós blokk
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 2. konvolúciós blokk
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # 3. konvolúciós blokk
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # GlobalAveragePooling2D: összesűríti a térbeli dimenziókat egyetlen vektorrá
        # A Flatten-nel szemben NEM hoz létre hatalmas teljesen összekötött réteget
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        # Osztályozó fej
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # AdamW: Adam + weight decay regularizáció a súlyok túlnövekedése ellen
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)

    # Label smoothing: a kemény [0,0,1,0,...] címkék helyett [0.014, 0.014, 0.914, 0.014, ...]
    # Ez megakadályozza, hogy a modell 99.99%-os magabiztossággal prediktáljon, ami overfittinghez vezet
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
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

# Label smoothing alkalmazása kézi one-hot kódolással
num_classes = len(CLASSES)
LABEL_SMOOTHING = 0.1
y_train_smooth = tf.one_hot(y_train, num_classes) * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
y_val_onehot = tf.one_hot(y_val, num_classes)
y_test_onehot = tf.one_hot(y_test, num_classes)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

model = build_improved_2d_cnn_model(
    (X_train.shape[1], X_train.shape[2], X_train.shape[3]),
    num_classes
)
model.summary()

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_model.keras'),
        monitor='val_accuracy', save_best_only=True
    ),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
]

print("\nTraining...")
history = model.fit(
    X_train, y_train_smooth,
    validation_data=(X_val, y_val_onehot),
    epochs=150,
    batch_size=32,
    callbacks=callbacks_list
)

# Predikció a teszthalmazon
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
plt.title(f'Confusion Matrix - {FEATURE_TYPE} 2D-CNN (Improved)')
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
sns.barplot(x=CLASSES, y=f1_scores, hue=CLASSES, palette='viridis', legend=False)
plt.title(f'F1-Scores per Class - {FEATURE_TYPE} (Improved)')
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
plt.title(f'Model Accuracy - {FEATURE_TYPE} (Improved)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f'Model Loss - {FEATURE_TYPE} (Improved)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
acc_loss_path = os.path.join(MODEL_SAVE_PATH, f'best_{FEATURE_TYPE}_2dcnn_acc_loss.png')
plt.savefig(acc_loss_path, dpi=300)
plt.show()
plt.close()
print(f"Accuracy and Loss plots saved successfully to {acc_loss_path}")
