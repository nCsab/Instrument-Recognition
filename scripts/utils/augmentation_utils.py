import numpy as np
import librosa
import random

def add_noise(audio, noise_path, sr=16000, duration=1.0, snr_db=10):
    """
    Környezeti zaj rákeverése a hangmintára.
    """
    try:
        noise, _ = librosa.load(noise_path, sr=sr, duration=duration)
        if len(noise) < len(audio):
            noise = np.pad(noise, (0, len(audio) - len(noise)))
        else:
            noise = noise[:len(audio)]
        
        p_audio = np.sum(audio ** 2) / len(audio)
        p_noise = np.sum(noise ** 2) / len(noise)
        snr = 10 ** (snr_db / 10)
        scale = np.sqrt(p_audio / (snr * p_noise + 1e-10))
        return audio + scale * noise
    except:
        return audio

def add_reverb(audio, sr=16000, delay_ms=30, decay=0.4):
    """
    Visszhang (reverberation) szimulálása.
    """
    try:
        delay_samples = int(sr * (delay_ms / 1000.0))
        taps = [delay_samples, int(delay_samples * 1.5), int(delay_samples * 2.1)]
        out = audio.copy()
        for i, t in enumerate(taps):
            if t < len(audio):
                out[t:] += audio[:-t] * (decay ** (i + 1))
        return out
    except:
        return audio

def add_eq(audio):
    """
    Egyszerű EQ/Pre-emphasis szűrő alkalmazása a frekvenciamenet változtatásához.
    """
    try:
        coef = random.uniform(-0.5, 0.5)
        return librosa.effects.preemphasis(audio, coef=coef)
    except:
        return audio

def add_pitch_shift(audio, sr=16000, n_steps=None):
    """
    Hangsúly eltolása (pitch shifting).
    """
    if n_steps is None:
        n_steps = random.uniform(-1.5, 1.5)
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except:
        return audio
