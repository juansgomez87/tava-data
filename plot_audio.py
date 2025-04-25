#!/usr/bin/env python3
"""
Audio plot for paper

Copyright 2025, J.S. Gómez-Cañón
Licensed under GNU AFFERO GENERAL PUBLIC LICENSE
"""


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import os
import numpy as np
import pandas as pd
import librosa

import pdb

def plot_audio():
    all_f = glob.glob(os.path.join(os.path.abspath('.'), 'audio', 'recordings', '*.wav'))
    idx = np.random.choice(len(all_f))
    this_aud = all_f[idx]
    this_egg = this_aud.replace('/recordings/', '/tegg/').replace('_audio', '_tegg')

    y_aud, sr = librosa.load(this_aud)
    y_egg, sr = librosa.load(this_egg)
    D_aud = librosa.stft(y_aud, n_fft=2048)  # STFT of y
    S_db_aud = librosa.amplitude_to_db(np.abs(D_aud), ref=np.max)
    D_egg = librosa.stft(y_egg, n_fft=2048)  # STFT of y
    S_db_egg = librosa.amplitude_to_db(np.abs(D_egg), ref=np.max)

    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4), constrained_layout=True, sharex='col')
    gs = gridspec.GridSpec(2, 3, width_ratios=[3, 3, 0.15], wspace=0.4, hspace=0.4)

    # Time axis
    time_aud = librosa.times_like(y_aud, sr=sr)
    time_egg = librosa.times_like(y_egg, sr=sr)

    axs[0, 0].plot(time_aud, y_aud, color='black', linewidth=0.8)
    axs[0, 0].set_title('Audio')
    axs[0, 0].set_ylabel('Amplitude')

    axs[1, 0].plot(time_egg, y_egg, color='gray', linewidth=0.8)
    axs[1, 0].set_title('tEGG')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Amplitude')

    # Plot spectrograms
    img1 = librosa.display.specshow(S_db_aud, sr=sr, x_axis='time', y_axis='log', ax=axs[0, 1], cmap='magma')
    axs[0, 1].set_title('Audio Spectrogram')

    img2 = librosa.display.specshow(S_db_egg, sr=sr, x_axis='time', y_axis='log', ax=axs[1, 1], cmap='magma')
    axs[1, 1].set_title('tEGG Spectrogram')
    axs[1, 1].set_xlabel('Time (s)')

    # Add colorbars to spectrograms
    # fig.colorbar(img1, ax=axs[0, 1], format='%+2.0f dB', shrink=0.7)
    # fig.colorbar(img2, ax=axs[1, 1], format='%+2.0f dB', shrink=0.7)

    # # Global title
    # if title:
    #     fig.suptitle(title, fontsize=12, y=1.02)

    # Tighten layout for publication
    plt.tight_layout()
    fig.savefig("figs/audio_tegg_plot.pdf", bbox_inches='tight', dpi=300)
    plt.show()
    pdb.set_trace()



if __name__ == '__main__':
    # plot audio and spectrograms
    plot_audio()