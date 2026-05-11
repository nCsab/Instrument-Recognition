import numpy as np
import librosa


def extract_log_mel(audio, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    return librosa.power_to_db(mel_spec, ref=np.max)


def extract_stft(audio, n_fft=1024, hop_length=512):
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
    return librosa.amplitude_to_db(np.abs(stft), ref=np.max)


def extract_mfcc(audio, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512):
    return librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )


def z_score_normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + 1e-10)
