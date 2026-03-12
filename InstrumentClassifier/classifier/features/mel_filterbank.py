import numpy as np
import librosa

from classifier import config


class MelFilterbankExtractor:

    def __init__(self, n_mels=config.N_MELS):
        self.n_mels = n_mels
        self.fft_size = config.FFT_SIZE
        self.hop_size = config.HOP_SIZE

    def extract(self, audio, sr):
        if audio.size == 0:
            raise ValueError("Audio array is empty.")

        mel_spec = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sr,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        mel_mean = np.mean(log_mel, axis=1)

        delta = librosa.feature.delta(log_mel)
        delta_mean = np.mean(delta, axis=1)

        delta2 = librosa.feature.delta(log_mel, order=2)
        delta2_mean = np.mean(delta2, axis=1)

        return np.concatenate([mel_mean, delta_mean, delta2_mean]).astype(np.float32)

    def get_feature_dim(self):
        return self.n_mels * 3
