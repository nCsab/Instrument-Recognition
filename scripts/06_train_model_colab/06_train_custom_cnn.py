import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import random

# --- Kísérleti Kontroll Beállítások ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

drive.mount('/content/drive')

PROJECT_ROOT = '/content/drive/MyDrive/Instrument_Recognition'
DATASET_TYPE = 'exp_clean'  # Pl.: 'exp_clean', 'exp_augmented', 'exp_naive_deployment', 'exp_final'
FEATURE_TYPE = 'log_mel'  # Lehetőségek: 'log_mel', 'stft', 'mfcc'
EVALUATION_SPLIT = 'val'  # Modellválasztáshoz: 'val'. Csak a végső győztesnél: 'test'.
TRAIN_MODEL = True  # False esetén a CHECKPOINT_TO_EVALUATE modellt tölti be újratanítás nélkül.
CHECKPOINT_TO_EVALUATE = None  # Pl.: 'exp_final_log_mel_2dcnn_val_20260527_203032_best_model.keras'

DATA_PATH = os.path.join(PROJECT_ROOT, 'processed_data', DATASET_TYPE)
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'models', DATASET_TYPE)
CLASSES = ["guitar", "piano", "vocal", "string", "reed", "brass", "noise"]


class SpecAugment(layers.Layer):
    def __init__(self, freq_mask_param=15, time_mask_param=8, num_masks=2, apply_freq_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks
        self.apply_freq_mask = apply_freq_mask

    def call(self, inputs, training=None):
        if not training:
            return inputs
        augmented = inputs
        freq_dim = tf.shape(inputs)[1]
        time_dim = tf.shape(inputs)[2]
        
        for _ in range(self.num_masks):
            # Frekvencia maszkolás (kivéve MFCC esetén)
            if self.apply_freq_mask:
                f = tf.random.uniform([], 1, self.freq_mask_param, dtype=tf.int32)
                f = tf.minimum(f, freq_dim)
                f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)
                indices = tf.range(freq_dim)
                freq_mask = tf.cast(tf.logical_or(indices < f0, indices >= f0 + f), tf.float32)
                augmented = augmented * tf.reshape(freq_mask, [1, -1, 1, 1])

            # Időbeli maszkolás (minden feature-nél alkalmazható)
            t = tf.random.uniform([], 1, self.time_mask_param, dtype=tf.int32)
            t = tf.minimum(t, time_dim)
            t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)
            indices_t = tf.range(time_dim)
            time_mask = tf.cast(tf.logical_or(indices_t < t0, indices_t >= t0 + t), tf.float32)
            augmented = augmented * tf.reshape(time_mask, [1, 1, -1, 1])
            
        return augmented

    def get_config(self):
        config = super().get_config()
        config.update({
            'freq_mask_param': self.freq_mask_param,
            'time_mask_param': self.time_mask_param,
            'num_masks': self.num_masks,
            'apply_freq_mask': self.apply_freq_mask
        })
        return config


def build_model(input_shape, num_classes, feature_type):
    # Log-Mel és STFT esetében alkalmazzuk a frekvencia-maszkolást, MFCC-nél kikapcsoljuk
    apply_freq_mask = False if feature_type == 'mfcc' else True
    
    model = models.Sequential([
        layers.Input(shape=input_shape),
        SpecAugment(freq_mask_param=15, time_mask_param=8, num_masks=2, apply_freq_mask=apply_freq_mask),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.3),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(), layers.ReLU(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def load_subset(subset, feature_type):
    X = np.load(os.path.join(DATA_PATH, f'X_{feature_type}_{subset}.npy'))
    if X.ndim == 3: X = X[..., np.newaxis]
    y = np.load(os.path.join(DATA_PATH, f'y_labels_{subset}.npy'))
    return X, y


def plot_results(y_true, y_pred, history, feature_type, eval_split, save_path, artifact_prefix):
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - {feature_type} ({eval_split})')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{artifact_prefix}_confusion_matrix.png'), dpi=300)
    plt.show(); plt.close()

    # F1 Scores
    f1_scores = [report_dict[cls]['f1-score'] for cls in CLASSES]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=CLASSES, y=f1_scores, hue=CLASSES, palette='viridis', legend=False)
    plt.title(f'F1-Scores - {feature_type} ({eval_split})')
    plt.ylabel('F1-Score'); plt.ylim(0, 1.1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{artifact_prefix}_f1_scores.png'), dpi=300)
    plt.show(); plt.close()

    if history is not None:
        # Accuracy & Loss
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
        plt.savefig(os.path.join(save_path, f'{artifact_prefix}_acc_loss.png'), dpi=300)
        plt.show(); plt.close()

    return report_dict


def save_evaluation_report(y_true, y_pred, report_text, report_dict, eval_split, save_path, artifact_prefix, checkpoint_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(save_path, f'{artifact_prefix}_confusion_matrix.csv')
    txt_path = os.path.join(save_path, f'{artifact_prefix}_classification_report.txt')
    json_path = os.path.join(save_path, f'{artifact_prefix}_classification_report.json')

    np.savetxt(cm_path, cm, fmt='%d', delimiter=',')

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {DATASET_TYPE}\n")
        f.write(f"Feature: {FEATURE_TYPE}\n")
        f.write(f"Evaluation split: {eval_split}\n")
        f.write("Model: 2dcnn\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(report_text)
        f.write(f"\nAccuracy: {report_dict['accuracy']*100:.1f}%\n")
        f.write(f"Macro F1: {report_dict['macro avg']['f1-score']*100:.1f}%\n")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": DATASET_TYPE,
            "feature": FEATURE_TYPE,
            "evaluation_split": eval_split,
            "model": "2dcnn",
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


def load_saved_model(checkpoint_path):
    return tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'SpecAugment': SpecAugment},
        compile=False
    )


# --- Main ---
def main():
    if EVALUATION_SPLIT not in ['val', 'test']:
        raise ValueError("EVALUATION_SPLIT must be 'val' or 'test'")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_prefix = f'{DATASET_TYPE}_{FEATURE_TYPE}_2dcnn_{EVALUATION_SPLIT}_{run_id}'
    checkpoint_path = resolve_checkpoint_path(CHECKPOINT_TO_EVALUATE)

    print(f"Loading data ({FEATURE_TYPE})...")
    if TRAIN_MODEL:
        X_train, y_train = load_subset('train', FEATURE_TYPE)
    X_val, y_val = load_subset('val', FEATURE_TYPE)
    if EVALUATION_SPLIT == 'test':
        X_eval, y_eval = load_subset('test', FEATURE_TYPE)
    else:
        X_eval, y_eval = X_val, y_val

    num_classes = len(CLASSES)
    
    # Szigorú Label Stratégia: One-Hot kódolás + Smoothing a tréningre
    LABEL_SMOOTHING = 0.1
    y_val_encoded = tf.one_hot(y_val, num_classes)

    print("Data shapes:")
    print(f"  train: {X_train.shape if TRAIN_MODEL else 'not loaded in evaluate-only mode'}")
    print(f"  val  : {X_val.shape}")
    if EVALUATION_SPLIT == 'test':
        print(f"  test : {X_eval.shape}\n")
    else:
        print("  test : not loaded during model selection\n")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    if TRAIN_MODEL:
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, f'{artifact_prefix}_best_model.keras')
        y_train_encoded = tf.one_hot(y_train, num_classes) * (1.0 - LABEL_SMOOTHING) + (LABEL_SMOOTHING / num_classes)
        model = build_model((X_train.shape[1], X_train.shape[2], X_train.shape[3]), num_classes, FEATURE_TYPE)
        model.summary()

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss', mode='min', save_best_only=True
            ),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]

        print(f"\nTraining... checkpoint: {checkpoint_path}")
        history = model.fit(
            X_train, y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=150, batch_size=32,
            callbacks=callbacks_list
        )
        model = load_saved_model(checkpoint_path)
    else:
        if checkpoint_path is None:
            raise ValueError("Set CHECKPOINT_TO_EVALUATE when TRAIN_MODEL is False.")
        print(f"\nLoading checkpoint without training: {checkpoint_path}")
        model = load_saved_model(checkpoint_path)
        history = None

    y_pred = np.argmax(model.predict(X_eval), axis=1)

    report_text = classification_report(y_eval, y_pred, target_names=CLASSES)

    print(f"\nClassification Report ({EVALUATION_SPLIT}):")
    print(report_text)

    report_dict = plot_results(y_eval, y_pred, history, FEATURE_TYPE, EVALUATION_SPLIT, MODEL_SAVE_PATH, artifact_prefix)
    save_evaluation_report(y_eval, y_pred, report_text, report_dict, EVALUATION_SPLIT, MODEL_SAVE_PATH, artifact_prefix, checkpoint_path)

    print(f"Accuracy: {report_dict['accuracy']*100:.1f}%")
    print(f"Macro F1: {report_dict['macro avg']['f1-score']*100:.1f}%")

if __name__ == "__main__":
    main()
