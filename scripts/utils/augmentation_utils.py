import os
import random
import numpy as np
import librosa


def add_noise(audio, noise_path, sr=16000, duration=1.0, snr_db=10):
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
    except Exception:
        return audio


def add_reverb(audio, sr=16000, delay_ms=30, decay=0.4):
    try:
        delay_samples = int(sr * (delay_ms / 1000.0))
        taps = [delay_samples, int(delay_samples * 1.5), int(delay_samples * 2.1)]
        out = audio.copy()
        for i, t in enumerate(taps):
            if t < len(audio):
                out[t:] += audio[:-t] * (decay ** (i + 1))
        return out
    except Exception:
        return audio


def add_eq(audio):
    try:
        coef = random.uniform(-0.5, 0.5)
        return librosa.effects.preemphasis(audio, coef=coef)
    except Exception:
        return audio


def add_pitch_shift(audio, sr=16000, n_steps=None):
    if n_steps is None:
        n_steps = random.uniform(-1.5, 1.5)
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except Exception:
        return audio


def _room_reverb(audio, sr=16000, delay_ms=30, decay=0.3):
    delay_samples = int(sr * (delay_ms / 1000.0))
    taps = [delay_samples, int(delay_samples * 1.5), int(delay_samples * 2.1), int(delay_samples * 2.8)]
    out = audio.copy()
    for i, t in enumerate(taps):
        if t < len(audio):
            out[t:] += audio[:-t] * (decay ** (i + 1))
    return out


def apply_macbook_augment(audio, noise_files, noise_path="", sr=16000):
    aug = audio.copy()

    if random.random() < 0.8:
        delay = random.uniform(15, 40)
        decay = random.uniform(0.2, 0.45)
        aug = _room_reverb(aug, sr=sr, delay_ms=delay, decay=decay)

    coef = random.uniform(0.92, 0.98)
    aug = librosa.effects.preemphasis(aug, coef=coef)
    aug = np.append(aug, 0.0)

    if len(noise_files) > 0:
        noise_file = random.choice(noise_files)
        try:
            duration = len(aug) / sr
            full_path = noise_file if os.path.isfile(noise_file) else os.path.join(noise_path, noise_file)
            noise_audio, _ = librosa.load(full_path, sr=sr, duration=duration + 1.0)

            if len(noise_audio) < len(aug):
                noise_audio = np.pad(noise_audio, (0, len(aug) - len(noise_audio)), mode="wrap")
            else:
                start = random.randint(0, len(noise_audio) - len(aug))
                noise_audio = noise_audio[start:start + len(aug)]

            snr_db = random.uniform(5.0, 15.0)
            p_audio = np.sum(aug ** 2) / len(aug)
            p_noise = np.sum(noise_audio ** 2) / len(noise_audio)
            snr = 10 ** (snr_db / 10.0)
            scale = np.sqrt(p_audio / (snr * p_noise + 1e-10))
            aug = aug + scale * noise_audio
        except Exception:
            pass

    drive = random.uniform(2.0, 4.0)
    aug = np.tanh(aug * drive) / np.tanh(drive)

    max_val = np.max(np.abs(aug))
    if max_val > 0.0:
        aug = (aug / max_val) * 0.9

    return aug
