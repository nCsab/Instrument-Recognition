import os
import shutil
import numpy as np
import librosa
import tensorflow as tf
from collections import Counter
from utils.feature_utils import extract_log_mel

# Konfiguráció
TEST_DIR = "/Volumes/Kingston XS1000 Media/project/model_test"
MODEL_PATH = "/Volumes/Kingston XS1000 Media/project/models/best_log_mel_model.keras"
CLASSES = ["guitar", "piano", "vocal", "string", "noise"]
SR = 16000

def build_standard_2d_cnn_model(input_shape, num_classes):
    from tensorflow.keras import layers, models
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
    return model

def normalize_features(feat):
    # Globális Min-Max (ahogy a tanításkor történt: [-80, 0] -> [0, 1])
    # A librosa.power_to_db(ref=np.max) mindig 0-ra teszi a peak-et.
    normalized = (feat - (-80.0)) / (0.0 - (-80.0) + 1e-10)
    return np.clip(normalized, 0.0, 1.0)

def split_audio_to_segments(file_path, output_dir, sr=16000, segment_duration=1.0):
    """Feldarabol egy hangfájlt 1 másodperces szegmensekre egy almappába."""
    y, file_sr = librosa.load(file_path, sr=sr)
    segment_samples = int(sr * segment_duration)
    
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for start in range(0, len(y) - segment_samples + 1, segment_samples):
        segment = y[start:start + segment_samples]
        segment_path = os.path.join(output_dir, f"segment_{count+1:03d}.wav")
        import soundfile as sf
        sf.write(segment_path, segment, sr)
        count += 1
    
    return count

def run_batch_prediction():
    print(f"Modell architektúra építése...")
    input_shape = (64, 32, 1)
    model = build_standard_2d_cnn_model(input_shape, len(CLASSES))
    
    print(f"Súlyok betöltése: {MODEL_PATH}...")
    try:
        model.load_weights(MODEL_PATH)
    except Exception as e:
        print(f"HIBA a súlyok betöltésekor: {e}")
        return

    # 1. Lépés: Keressünk hangfájlokat közvetlenül a model_test mappában
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')
    direct_files = [f for f in os.listdir(TEST_DIR) 
                    if os.path.isfile(os.path.join(TEST_DIR, f)) and f.lower().endswith(audio_extensions)]
    
    if direct_files:
        print(f"\n🔪 {len(direct_files)} hangfájl találva a mappában, feldarabolás 1s szegmensekre...\n")
        for f in direct_files:
            file_path = os.path.join(TEST_DIR, f)
            folder_name = os.path.splitext(f)[0]
            output_dir = os.path.join(TEST_DIR, folder_name)
            
            # Ha már létezik a mappa, töröljük és újrageneráljuk
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            seg_count = split_audio_to_segments(file_path, output_dir, sr=SR)
            print(f"  ✅ {f} → {seg_count} szegmens ({folder_name}/)")

    # 2. Lépés: Keressük az al-mappákat (beleértve az újonnan létrehozottakat)
    subdirs = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    
    if not subdirs:
        print("Nem találtam feldolgozandó fájlokat vagy almappákat!")
        return

    for subdir in subdirs:
        subdir_path = os.path.join(TEST_DIR, subdir)
        files = [f for f in os.listdir(subdir_path) if f.endswith(".wav")]
        
        if not files:
            continue
            
        print(f"\n--- OSZTÁLYOZÁS: {subdir} ---")
        
        # Próbáljuk kitalálni az elvárt osztályt a mappa nevéből
        expected_class = None
        lower_subdir = subdir.lower()
        if "guitar" in lower_subdir: expected_class = "guitar"
        elif "piano" in lower_subdir: expected_class = "piano"
        elif "vocal" in lower_subdir: expected_class = "vocal"
        elif "string" in lower_subdir: expected_class = "string"
        elif "noise" in lower_subdir: expected_class = "noise"
        
        if expected_class:
            print(f"Észlelt elvárt osztály: {expected_class.upper()}")
            # Töröljük a korábbi hibákat ha voltak
            mis_dir = os.path.join(subdir_path, "misclassified")
            if os.path.exists(mis_dir):
                shutil.rmtree(mis_dir)
        
        print(f"Szegmensek száma: {len(files)}")
        predictions = []
        
        for i, f in enumerate(sorted(files)):
            file_path = os.path.join(subdir_path, f)
            try:
                y, _ = librosa.load(file_path, sr=SR)
                if len(y) < SR: continue
                
                feat_raw = extract_log_mel(y[:SR])
                feat = normalize_features(feat_raw)
                
                X = feat.reshape(1, feat.shape[0], feat.shape[1], 1)
                
                pred_prob = model.predict(X, verbose=0)
                pred_idx = np.argmax(pred_prob)
                predicted_class = CLASSES[pred_idx]
                predictions.append(predicted_class)
                
                # Ha nem az elvártat prediktálta, mentsük ki
                if expected_class and predicted_class != expected_class:
                    target_mis_dir = os.path.join(subdir_path, "misclassified", predicted_class)
                    os.makedirs(target_mis_dir, exist_ok=True)
                    shutil.copy(file_path, os.path.join(target_mis_dir, f))
                
            except Exception as e:
                continue

        # Összegzés
        if predictions:
            counts = Counter(predictions)
            total = len(predictions)
            print(f"\nEredmények összefoglalása ({subdir}):")
            for cls in CLASSES:
                count = counts.get(cls, 0)
                percentage = (count / total) * 100
                print(f"  {cls:8}: {percentage:5.1f}% ({count} db)")
            
            if expected_class:
                print(f"\nA tévesztett fájlok átmásolva ide: {subdir}/misclassified/")
        else:
            print("Nem sikerült jóslatokat készíteni ebben a mappában.")

if __name__ == "__main__":
    run_batch_prediction()

