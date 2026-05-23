import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

audio_path = '/Volumes/Kingston XS1000 Media/project/piano_chord.mp3'
out_file = '/Volumes/Kingston XS1000 Media/project/thesis/fig_spectrogram_example.png'

print(f"Loading {audio_path}...")
y, sr = librosa.load(audio_path, sr=22050)
y_trimmed, _ = librosa.effects.trim(y, top_db=40)

# Compute STFT
D = librosa.stft(y_trimmed, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB', label='Energia (dB)')
plt.title('Zongoraakkord STFT Spektrogramja', fontweight='bold', pad=15)
plt.xlabel('Idő (s)')
plt.ylabel('Frekvencia (Hz)')
plt.ylim(0, 8000) # Limit frequency to 8 kHz for better visibility of the chord
plt.tight_layout()

plt.savefig(out_file, dpi=300)
print(f"Saved {out_file}")
