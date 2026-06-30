import numpy as np
import librosa

# Közös feature extraction segédfüggvények.
# A tanítási és realtime kód ugyanazokat a beállításokat használja, hogy a
# modell bemenete konzisztens maradjon.

DB_MIN = -80.0
DB_MAX = 0.0


def extract_log_mel(audio, sr=16000, n_mels=128, n_fft=1024, hop_length=256):
    # Perceptuális, Mel-skálájú spektrogram dB skálán.
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def extract_stft(audio, n_fft=1024, hop_length=512):
    # Klasszikus idő-frekvencia spektrogram dB skálán.
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)


def extract_mfcc(audio, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512):
    # Tömör, kepsztrális audiojellemző baseline-ként.
    return librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )


def z_score_normalize(data):
    # MFCC esetén használt csatornánkénti standardizálás.
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + 1e-10)


def normalize_db_feature(data, db_min=DB_MIN, db_max=DB_MAX):
    # Log-Mel/STFT esetén fix dB-tartományból [0, 1] skálázás.
    data = np.clip(data, db_min, db_max)
    return (data - db_min) / (db_max - db_min + 1e-10)
