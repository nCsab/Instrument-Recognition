
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

DATA_PATH = "/Volumes/Kingston XS1000 Media/project/processed_data"
MODEL_SAVE_PATH = "/Volumes/Kingston XS1000 Media/project/models"
CLASSES = ["guitar", "piano", "other", "noise"]

BATCH_SIZE = 32
EPOCHS = 50
INPUT_SHAPE = (64, 44, 1)

def load_data():
    print("Loading data...")
    X = np.load(os.path.join(DATA_PATH, "X_hybrid.npy"))
    y = np.load(os.path.join(DATA_PATH, "y_hybrid.npy"))
    
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    x_min = X.min()
    x_max = X.max()
    X = (X - x_min) / (x_max - x_min + 1e-10)
    
    print(f"Data Loaded. X shape: {X.shape}, labels: {y.shape}")
    print(f"Range: [{X.min():.2f}, {X.max():.2f}]")
    
    return X, y

def build_model():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
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
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train():
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = build_model()
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, "instrument_model_best.keras"), 
        monitor='val_accuracy', 
        save_best_only=True
    )

    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint]
    )
    
    model.save(os.path.join(MODEL_SAVE_PATH, "instrument_model_final.keras"))
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

