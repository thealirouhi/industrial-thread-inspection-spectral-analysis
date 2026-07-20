import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows
import os


def generate_demo_audio(fs=48000, duration=2.0):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f1, f2, f3 = 200, 400, 600
    segment_len = len(t) // 3
    s1 = np.sin(2 * np.pi * f1 * t[:segment_len])
    s2 = np.sin(2 * np.pi * f2 * t[segment_len:2*segment_len])
    s3 = np.sin(2 * np.pi * f3 * t[2*segment_len:])
    signal = np.concatenate([s1, s2, s3])
    signal += 0.3 * np.sin(2 * np.pi * 1.5 * f1 * t)
    return signal, t, fs


def plot_spectrogram(signal, fs, nperseg, noverlap, window, ax, title=""):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, window=window,
                                  nperseg=nperseg, noverlap=noverlap)
    ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim([0, fs / 2])
    return f, t_spec, Sxx


def study_window_type(output_dir):
    signal, t, fs = generate_demo_audio()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    windows_to_test = ["hann", "hamming", "blackman", "boxcar", "flattop", ("tukey", 0.5)]

    for idx, win in enumerate(windows_to_test):
        row, col = divmod(idx, 3)
        win_name = win if isinstance(win, str) else f"tukey(0.5)"
        plot_spectrogram(signal, fs, nperseg=1024, noverlap=768,
                         window=win, ax=axes[row, col],
                         title=f"Window: {win_name}")

    plt.suptitle("Effect of Window Type on Spectrogram", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrogram_window_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spectrogram_window_comparison.png")


def study_nperseg(output_dir):
    signal, t, fs = generate_demo_audio()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    nperseg_values = [256, 1024, 4096]

    for idx, nperseg in enumerate(nperseg_values):
        plot_spectrogram(signal, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4,
                         window="hann", ax=axes[idx],
                         title=f"nperseg={nperseg}\n(freq res={fs/nperseg:.1f} Hz)")

    plt.suptitle("Effect of Window Size (nperseg) on Spectrogram", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrogram_nperseg_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spectrogram_nperseg_comparison.png")


def study_overlap(output_dir):
    signal, t, fs = generate_demo_audio()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    overlap_values = [0, 256, 768]
    nperseg = 1024

    for idx, noverlap in enumerate(overlap_values):
        plot_spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap,
                         window="hann", ax=axes[idx],
                         title=f"noverlap={noverlap}\n(overlap={noverlap/nperseg*100:.0f}%)")

    plt.suptitle("Effect of Overlap on Spectrogram", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrogram_overlap_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spectrogram_overlap_comparison.png")


def study_sampling_freq(output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fs_values = [8000, 48000, 96000]

    for idx, fs in enumerate(fs_values):
        signal, t, fs_actual = generate_demo_audio(fs=fs, duration=2.0)
        plot_spectrogram(signal, fs, nperseg=1024, noverlap=768,
                         window="hann", ax=axes[idx],
                         title=f"fs={fs} Hz\n(max freq={fs//2} Hz)")

    plt.suptitle("Effect of Sampling Frequency on Spectrogram", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrogram_fs_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spectrogram_fs_comparison.png")


def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 2, Step 1: Spectrogram Visualization & Parameter Study ===\n")
    print("Using generated demo audio (chirp-like signal)\n")

    print("A) Comparing window types...")
    study_window_type(output_dir)

    print("B) Comparing window sizes (nperseg)...")
    study_nperseg(output_dir)

    print("C) Comparing overlap values...")
    study_overlap(output_dir)

    print("D) Comparing sampling frequencies...")
    study_sampling_freq(output_dir)

    print("\nDone! All figures saved to figures/")


if __name__ == "__main__":
    main()
