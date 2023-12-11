import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

sampling_rate = 44100  

piano_waveform = np.loadtxt('piano.txt')
trumpet_waveform = np.loadtxt('trumpet.txt')

# TODO: PART A
def plot_waveform_fft(waveform, title):
    t = np.arange(len(waveform)) / sampling_rate

    fft_result = fft(waveform)
    freq = np.fft.fftfreq(len(fft_result), 1/sampling_rate)[:10000]

    magnitudes = np.abs(fft_result)[:10000]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax[0].plot(t, waveform)
    ax[0].set_title(f"{title} Waveform")
    ax[0].set_xlabel("Time (seconds)")
    ax[0].set_ylabel("Amplitude")
  
    ax[1].plot(freq, magnitudes)
    ax[1].set_title(f"{title} Fourier Transform")
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude")

    plt.tight_layout()
    plt.show()
    return

plot_waveform_fft(piano_waveform, "Piano")
plot_waveform_fft(trumpet_waveform, "Trumpet")

# TODO: PART B

def find_fund_freq(waveform):
    N = len(waveform)

    T = 1.0 / 44100.0

    yf = fft(waveform)
    fundamental_freq_index = np.argmax(np.abs(yf[:N//2]))
    fundamental_frequency = fundamental_freq_index / (N * T)

    return fundamental_frequency

piano_fund_freq = find_fund_freq(piano_waveform)
trumpet_fund_freq = find_fund_freq(trumpet_waveform)

piano_fund_freq, trumpet_fund_freq
