import os
import threading
import numpy as np
import tensorflow as tf
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sounddevice as sd


class MonophonicInstrumentClassifier:
    def __init__(self, data_dir='dataset'):
        self.data_dir = Path(data_dir)
        self.sample_rate = 22050
        self.duration = 3.0 
        self.n_mfcc = 13
        self.max_pad_len = 130
        self.model = None

    def extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            if mfccs.shape[1] < self.max_pad_len:
                pad_width = self.max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfccs = mfccs[:, :self.max_pad_len]
            return mfccs.T.astype(np.float32)
        except Exception as e:
            print(f"Hiba a jellemzők kinyerésében {file_path}: {e}")
            return None

    def load_dataset(self):
        X, y, files = [], [], []
        instruments = ['piano', 'violin']
        for label, instrument in enumerate(instruments):
            folder = self.data_dir / instrument
            if not folder.exists():
                print(f"Nem található mappa: {folder}")
                continue
            audio_files = []
            for ext in ['wav', 'aif', 'aiff']:
                audio_files.extend(folder.glob(f"*.{ext}"))
            print(f"{instrument.capitalize()} minták: {len(audio_files)}")
            for file in audio_files:
                features = self.extract_features(file)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    files.append(str(file))
                    if len(X) % 10 == 0:
                        print(f"Feldolgozott minták: {len(X)}")
        if len(X) == 0:
            raise ValueError("Nincs feldolgozható adat!")
        return np.array(X), np.array(y), np.array(files)

    def create_model(self, input_shape):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        model.summary()
        return model

    def _play_test_files(self, files_test, y_test, y_pred, class_names):
        import librosa
        for f, yt, yp in zip(files_test, y_test, y_pred):
            print(f"\nFájl: {f}")
            print(f"Valódi: {class_names[yt]} | Predikció: {class_names[yp]}")
            try:
                audio, sr = librosa.load(f, sr=None, duration=2.0)
                try:
                    import IPython.display as ipd
                    display(ipd.Audio(audio, rate=sr))
                except Exception:
                    sd.play(audio, sr)
                    sd.wait()
            except Exception as e:
                print(f"(Nem sikerült lejátszani: {e})")

    def train(self, X, y, files):
        X_train, X_temp, y_train, y_temp, files_train, files_temp = train_test_split(
            X, y, files, test_size=0.3, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test, files_val, files_test = train_test_split(
            X_temp, y_temp, files_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"Tanító minták: {len(y_train)}, Validációs: {len(y_val)}, Teszt: {len(y_test)}")

        self.model = self.create_model(input_shape=X_train.shape[1:])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)
        ]

        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=callbacks
        )

        print("\nTeszt értékelés:")
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Teszt pontosság: {test_acc:.4f}")

        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        labels = unique_labels(y_test, y_pred)
        class_names = ['Piano', 'Violin']
        target_names = [class_names[i] for i in labels]

        print("\nRészletes jelentés:")
        print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predikciók')
        plt.ylabel('Valódi osztályok')
        plt.title('Confusion matrix')
        plt.savefig("confusion_matrix.png")
        plt.close()

        thread = threading.Thread(target=self._play_test_files, args=(files_test, y_test, y_pred, class_names))
        thread.start()

    def predict(self, audio_path):
        features = self.extract_features(audio_path)
        if features is None or self.model is None:
            print("Nincs betanított modell vagy hiba a fájlban!")
            return None
        pred = self.model.predict(np.expand_dims(features, axis=0))
        idx = np.argmax(pred[0])
        classes = ['Piano', 'Violin']
        print(f"Predikált hangszer: {classes[idx]} (bizonyosság: {pred[0][idx]:.2f})")
        return classes[idx], pred[0][idx]


def main():
    classifier = MonophonicInstrumentClassifier(data_dir='dataset')
    X, y, files = classifier.load_dataset()
    classifier.train(X, y, files)

    print("\nPélda előrejelzés egy fájlon...")
    sample_file = Path('dataset/piano/sample1.wav')
    if sample_file.exists():
        classifier.predict(str(sample_file))
    else:
        print("Nincs példa fájl a megadott helyen.")


if __name__ == "__main__":
    main()
