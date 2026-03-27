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

def add_room_reverb(audio, sr=16000, delay_ms=30, decay=0.3):
    """Sűrűbb visszhang a kis szoba szimulálására."""
    delay_samples = int(sr * (delay_ms / 1000.0))
    taps = [delay_samples, int(delay_samples * 1.5), int(delay_samples * 2.1), int(delay_samples * 2.8)]
    out = audio.copy()
    for i, t in enumerate(taps):
        if t < len(audio):
            out[t:] += audio[:-t] * (decay ** (i + 1))
    return out

import os

def apply_macbook_augment(
    audio,
    noise_files,
    noise_path="/Volumes/Kingston XS1000 Media/project/hybrid_dataset/noise",
    sr=16000,
    p_apply=0.7,
):
    """
    Életszerű MacBook mic szimuláció:
    - Néha teljesen tisztán hagyjuk (1 - p_apply)
    - Enyhe szobareverb
    - Laptop-mikrofon jellegű pre-emphasis (0.85–0.95)
    - Mérsékelt háttérzaj (10–20 dB SNR)
    - Soft clipping tanh-hal az AGC szimulálására
    """
    aug = audio.copy()

    # 0. Néha egyáltalán NE augmentáljunk (maradjon stúdió-minőség)
    if random.random() > p_apply:
        return aug

    # 1. Enyhe Reverb (szoba/iroda akusztika, max 50% eséllyel)
    if random.random() < 0.5:
        delay = random.uniform(10, 30)    # rövidebb, valóságosabb fal-visszaverődés
        decay = random.uniform(0.1, 0.3)  # gyorsabb lecsengés
        aug = add_room_reverb(aug, sr=sr, delay_ms=delay, decay=decay)

    # 2. EQ / "laptop mikrofonsáv" (70% esély)
    if random.random() < 0.7:
        # Tipikus pre-emphasis tartomány (0.85–0.95) → magasak emelése, mélyek vágása
        coef = random.uniform(0.85, 0.95)
        aug = librosa.effects.preemphasis(aug, coef=coef)
        # Pre-emphasis 1 mintával rövidíti a jelet → kipótoljuk
        aug = np.append(aug, 0.0)

    # 3. Háttérzaj (10–20 dB SNR, 70% eséllyel)
    if random.random() < 0.7 and len(noise_files) > 0:
        noise_file = random.choice(noise_files)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                duration = len(aug) / sr
                full_path = noise_file if os.path.isfile(noise_file) else os.path.join(noise_path, noise_file)
                noise_audio, _ = librosa.load(full_path, sr=sr, duration=duration + 1.0)

            # Méret igazítása
            if len(noise_audio) < len(aug):
                noise_audio = np.pad(noise_audio, (0, len(aug) - len(noise_audio)), mode="wrap")
            else:
                start = random.randint(0, len(noise_audio) - len(aug))
                noise_audio = noise_audio[start:start + len(aug)]

            # 10–20 dB SNR → hangszer domináns, de zaj jól hallható
            snr_db = random.uniform(10.0, 20.0)
            p_audio = np.sum(aug ** 2) / len(aug)
            p_noise = np.sum(noise_audio ** 2) / len(noise_audio)
            snr = 10 ** (snr_db / 10.0)
            scale = np.sqrt(p_audio / (snr * p_noise + 1e-10))
            aug = aug + scale * noise_audio
        except Exception:
            pass

    # 4. Lágy kompresszor (Soft Clipping tanh-hal, 80% eséllyel)
    if random.random() < 0.8:
        drive = random.uniform(1.5, 3.0)
        aug = np.tanh(aug * drive) / np.tanh(drive)

    # 5. Clippelésvédelem + változatos hangerő
    max_val = np.max(np.abs(aug))
    if max_val > 0.0:
        target_peak = random.uniform(0.7, 0.95)
        aug = (aug / max_val) * target_peak

    return aug
