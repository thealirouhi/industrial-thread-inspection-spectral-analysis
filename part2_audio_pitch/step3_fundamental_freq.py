import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os


def generate_chirp_signal(fs=48000, duration=3.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f_start, f_end = 100, 500
    freq = f_start + (f_end - f_start) * t / duration
    signal = np.sin(2 * np.pi * np.cumsum(freq) / fs)
    return signal, t, fs, freq


def generate_harmonic_signal(fs=48000, duration=3.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f0 = 150 + 50 * np.sin(2 * np.pi * 0.5 * t)
    signal = (np.sin(2 * np.pi * f0 * t) +
              0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
              0.3 * np.sin(2 * np.pi * 3 * f0 * t))
    return signal, t, fs, f0


def extract_fundamental_freq(signal, fs, nperseg=1024, noverlap=768):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, window="hann",
                                  nperseg=nperseg, noverlap=noverlap)
    f0_per_frame = np.zeros(len(t_spec))
    for i in range(Sxx.shape[1]):
        col = Sxx[:, i]
        search_mask = f > 50
        f_search = f[search_mask]
        col_search = col[search_mask]
        peak_idx = np.argmax(col_search)
        f0_per_frame[i] = f_search[peak_idx]
    return t_spec, f0_per_frame, f, t_spec, Sxx


def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 2, Step 3: Fundamental Frequency f0(t) Extraction ===\n")

    signals = [
        ("Chirp Signal (100->500 Hz)", *generate_chirp_signal()),
        ("Harmonic Signal (f0 varying)", *generate_harmonic_signal()),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    for idx, (name, signal, t, fs, true_freq) in enumerate(signals):
        t_spec, f0, f, t_spec_full, Sxx = extract_fundamental_freq(signal, fs)

        ax_spec = axes[idx * 2]
        ax_spec.pcolormesh(t_spec_full, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_title(f"{name} - Spectrogram")
        ax_spec.set_ylim([0, 800])

        ax_f0 = axes[idx * 2 + 1]
        ax_f0.plot(t_spec, f0, "b-", linewidth=2, label="Extracted f0(t)")
        if len(true_freq) == len(t_spec):
            ax_f0.plot(t_spec, true_freq[:len(t_spec)], "r--", linewidth=1.5, label="True f0(t)")
        ax_f0.set_ylabel("Frequency (Hz)")
        ax_f0.set_xlabel("Time (s)")
        ax_f0.set_title(f"{name} - Fundamental Frequency f0(t)")
        ax_f0.legend()
        ax_f0.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fundamental_frequency_extraction.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fundamental_frequency_extraction.png")

    print("\nDone! Figure saved to figures/")


if __name__ == "__main__":
    main()
