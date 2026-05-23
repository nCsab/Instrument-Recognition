import os
import librosa
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

base_dir = '/Volumes/Kingston XS1000 Media/project/asd'
thesis_dir = '/Volumes/Kingston XS1000 Media/project/thesis'
zoom_duration = 0.005 # 5 ms

orders = [
    ('brass', ['tuba.wav', 'trombone.wav', 'horn.wav', 'trumpet.wav']),
    ('reed', ['bassoon.wav', 'sax.wav', 'bbclarinet.wav', 'oboe.wav']),
    ('string', ['contrabass.wav', 'violoncello.wav', 'viola.wav', 'violin.wav']),
    ('.', ['keyboard_acoustic_007-071-075.wav', 'guitar_acoustic_016-059-100.wav', 'vocal_acoustic_024-054-025.wav'])
]

fig = plt.figure(figsize=(18, 16))

def plot_wave(ax, file_path, title):
    if not os.path.exists(file_path):
        ax.set_visible(False)
        return
        
    y, sr = librosa.load(file_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    
    if len(y_trimmed) == 0:
        y_trimmed = y
        
    # Detektáljuk a fundamentális frekvenciát és toljuk el 440 Hz-re (A4)
    f0 = librosa.yin(y_trimmed, fmin=50, fmax=2000, sr=sr)
    valid_f0 = f0[f0 > 0]
    if len(valid_f0) > 0:
        current_f0 = np.median(valid_f0)
        target_f0 = 440.0
        n_steps = 12 * np.log2(target_f0 / current_f0)
        if abs(n_steps) > 0.1:
            y_trimmed = librosa.effects.pitch_shift(y_trimmed, sr=sr, n_steps=n_steps)
            
    if 'acoustic' in file_path and 'vocal' not in file_path:
        peak_sample = np.argmax(np.abs(y_trimmed))
        center = peak_sample + int(0.05 * sr)
        if center >= len(y_trimmed):
            center = peak_sample
    else:
        center = len(y_trimmed) // 2
        
    half_window = int((zoom_duration * sr) / 2)
    start_sample = max(0, center - half_window)
    end_sample = min(len(y_trimmed), center + half_window)
    
    y_zoom = y_trimmed[start_sample:end_sample]
    t_zoom = np.linspace(0, zoom_duration, len(y_zoom)) * 1000
    
    ax.plot(t_zoom, y_zoom, color='#1f77b4', lw=2.0)
    
    # Clean up titles
    clean_title = title.replace('.wav', '')
    clean_title = clean_title.replace('keyboard_acoustic_007-071-075', 'Zongora')
    clean_title = clean_title.replace('guitar_acoustic_016-059-100', 'Gitár')
    clean_title = clean_title.replace('vocal_acoustic_024-054-025', 'Ének (Vocal)')
    clean_title = clean_title.capitalize()
    
    ax.set_title(clean_title, fontweight='bold', pad=10)
    
    max_amp = np.max(np.abs(y_zoom)) if len(y_zoom) > 0 else 1.0
    ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)

for row_idx, (cat, files) in enumerate(orders):
    cat_dir = os.path.join(base_dir, cat)
    
    if len(files) == 4:
        for col_idx, file in enumerate(files):
            ax = fig.add_subplot(4, 4, row_idx * 4 + col_idx + 1)
            plot_wave(ax, os.path.join(cat_dir, file), file)
            if col_idx == 0:
                ax.set_ylabel('Amplitúdó')
            ax.set_xlabel('Idő (ms)')
    elif len(files) == 3:
        for col_idx, file in enumerate(files):
            ax = fig.add_subplot(4, 3, row_idx * 3 + col_idx + 1)
            plot_wave(ax, os.path.join(cat_dir, file), file)
            if col_idx == 0:
                ax.set_ylabel('Amplitúdó')
            ax.set_xlabel('Idő (ms)')

out_file = os.path.join(thesis_dir, 'fig_all_waveforms_combined.png')
plt.subplots_adjust(hspace=0.6, wspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Saved {out_file}")
