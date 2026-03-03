import numpy as np
import librosa

from classifier import config


class MFCCFeatureExtractor:

    def __init__(self):
        self.n_mfcc = config.N_MFCC
        self.n_mels = config.N_MELS
        self.fft_size = config.FFT_SIZE
        self.hop_size = config.HOP_SIZE

    def extract(self, audio, sr):
        if audio.size == 0:
            raise ValueError("Audio array is empty.")
        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1D, got shape {audio.shape}.")

        mfccs = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
        )

        mfcc_mean = np.mean(mfccs, axis=1)

        delta = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta, axis=1)

        delta2 = librosa.feature.delta(mfccs, order=2)
        delta2_mean = np.mean(delta2, axis=1)

        return np.concatenate([mfcc_mean, delta_mean, delta2_mean]).astype(np.float32)

    def get_feature_dim(self):
        return self.n_mfcc * 3
