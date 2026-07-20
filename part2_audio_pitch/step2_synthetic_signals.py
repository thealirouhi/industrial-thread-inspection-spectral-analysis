import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os


def generate_sinusoidal_signals(fs=48000, duration_each=1.0):
    t = np.linspace(0, duration_each, int(fs * duration_each), endpoint=False)
    f1, f2, f3 = 100, 150, 180
    s1 = np.sin(2 * np.pi * f1 * t)
    s2 = np.sin(2 * np.pi * f2 * t)
    s3 = np.sin(2 * np.pi * f3 * t)
    signal = np.concatenate([s1, s2, s3])
    total_t = np.linspace(0, 3 * duration_each, len(signal), endpoint=False)
    return signal, total_t, fs, [f1, f2, f3]


def generate_square_signals(fs=48000, duration_each=1.0):
    t = np.linspace(0, duration_each, int(fs * duration_each), endpoint=False)
    f1, f2, f3 = 100, 150, 180
    s1 = np.sign(np.sin(2 * np.pi * f1 * t))
    s2 = np.sign(np.sin(2 * np.pi * f2 * t))
    s3 = np.sign(np.sin(2 * np.pi * f3 * t))
    signal = np.concatenate([s1, s2, s3])
    total_t = np.linspace(0, 3 * duration_each, len(signal), endpoint=False)
    return signal, total_t, fs, [f1, f2, f3]


def generate_triangle_signals(fs=48000, duration_each=1.0):
    t = np.linspace(0, duration_each, int(fs * duration_each), endpoint=False)
    f1, f2, f3 = 100, 150, 180
    s1 = 2 * np.abs(2 * (f1 * t - np.floor(f1 * t + 0.5))) - 1
    s2 = 2 * np.abs(2 * (f2 * t - np.floor(f2 * t + 0.5))) - 1
    s3 = 2 * np.abs(2 * (f3 * t - np.floor(f3 * t + 0.5))) - 1
    signal = np.concatenate([s1, s2, s3])
    total_t = np.linspace(0, 3 * duration_each, len(signal), endpoint=False)
    return signal, total_t, fs, [f1, f2, f3]


def plot_spectrogram_and_fft(signal, t, fs, freqs, title, ax_spec, ax_fft):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, window="hann",
                                  nperseg=1024, noverlap=768)
    ax_spec.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_title(f"{title} - Spectrogram")
    ax_spec.set_ylim([0, 600])

    for fi in freqs:
        ax_spec.axhline(y=fi, color="white", linestyle="--", alpha=0.5, linewidth=0.5)

    N = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freqs = np.fft.rfftfreq(N, d=1/fs)
    ax_fft.plot(fft_freqs, fft_vals / np.max(fft_vals), "b-", linewidth=0.8)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Normalized Magnitude")
    ax_fft.set_title(f"{title} - FFT")
    ax_fft.set_xlim([0, 600])
    ax_fft.grid(True, alpha=0.3)

    for fi in freqs:
        ax_fft.axvline(x=fi, color="r", linestyle="--", alpha=0.5, linewidth=0.5)


def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 2, Step 2: Synthetic Signal Spectrograms ===\n")

    generators = [
        ("Sinusoidal", generate_sinusoidal_signals),
        ("Square Wave", generate_square_signals),
        ("Triangle Wave", generate_triangle_signals),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 15))

    for row, (name, gen_func) in enumerate(generators):
        signal, t, fs, freqs = gen_func()
        plot_spectrogram_and_fft(signal, t, fs, freqs, name,
                                  axes[row, 0], axes[row, 1])

    plt.suptitle("Spectrogram vs FFT for Different Waveforms\n(100Hz, 150Hz, 180Hz concatenated)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "synthetic_spectrogram_vs_fft.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved synthetic_spectrogram_vs_fft.png")

    print("\nDone! Figure saved to figures/")


if __name__ == "__main__":
    main()
