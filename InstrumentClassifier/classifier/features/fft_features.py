import numpy as np
from classifier import config


class FFTFeatureExtractor:

    def __init__(self):
        self.fft_size = config.FFT_SIZE
        self.hop_size = config.HOP_SIZE
        self.sr = config.SAMPLE_RATE
        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.freqs = np.fft.rfftfreq(self.fft_size, d=1.0 / self.sr)

    def extract(self, audio, sr):
        if audio.size == 0:
            raise ValueError("Audio array is empty.")

        frames = self._frame_audio(audio)
        if len(frames) == 0:
            padded = np.zeros(self.fft_size, dtype=np.float32)
            padded[:len(audio)] = audio
            frames = [padded]

        frame_features = []
        for frame in frames:
            features = self._extract_frame_features(frame)
            frame_features.append(features)

        return np.mean(frame_features, axis=0).astype(np.float32)

    def _frame_audio(self, audio):
        frames = []
        start = 0
        while start + self.fft_size <= len(audio):
            frame = audio[start:start + self.fft_size].astype(np.float32)
            frame = frame * self.window
            frames.append(frame)
            start += self.hop_size
        return frames

    def _extract_frame_features(self, frame):
        spectrum = np.fft.rfft(frame)
        magnitude = np.abs(spectrum).astype(np.float32)

        mag_sum = np.sum(magnitude)
        if mag_sum == 0:
            return np.zeros(self.get_feature_dim(), dtype=np.float32)

        mag_norm = magnitude / mag_sum
        features = []

        dominant_idx = np.argmax(magnitude)
        features.append(self.freqs[dominant_idx])
        features.append(magnitude[dominant_idx] / mag_sum)

        centroid = float(np.sum(self.freqs * mag_norm))
        features.append(centroid)

        bandwidth = float(np.sqrt(np.sum(mag_norm * (self.freqs - centroid) ** 2)))
        features.append(bandwidth)

        cumsum = np.cumsum(mag_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.85)
        rolloff_idx = min(rolloff_idx, len(self.freqs) - 1)
        features.append(float(self.freqs[rolloff_idx]))

        mag_pos = magnitude[magnitude > 0]
        if len(mag_pos) > 0:
            geo_mean = np.exp(np.mean(np.log(mag_pos + 1e-10)))
            arith_mean = np.mean(mag_pos)
            features.append(float(geo_mean / (arith_mean + 1e-10)))
        else:
            features.append(0.0)

        for h in range(2, 7):
            harmonic_idx = dominant_idx * h
            if harmonic_idx < len(magnitude):
                features.append(float(magnitude[harmonic_idx] / (magnitude[dominant_idx] + 1e-10)))
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def get_feature_dim(self):
        return 2 + 4 + 5
