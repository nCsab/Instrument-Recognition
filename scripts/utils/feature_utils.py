import numpy as np
import librosa

def extract_log_mel(audio, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel

def extract_stft(audio, n_fft=1024, hop_length=512):
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    # Use power spectrogram (magnitude squared) then to dB for consistency
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    # Ensure it's 2D and has similar scale to others
    return stft_db

def extract_mfcc(audio, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc

def z_score_normalize(data):
    """
    Z-score normalizálás (CMVN): koefficsiensek szerinti (axis=1) standardizálás.
    Így minden MFCC-sáv külön átlag/szórás skálát kap.
    """
    # data shape: (n_mfcc, T)
    mean = np.mean(data, axis=1, keepdims=True)
    std  = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + 1e-10)
