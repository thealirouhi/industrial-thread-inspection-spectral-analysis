import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, windows
from scipy.io import wavfile
import os


def load_recorded_audio(data_dir):
    wav_path = os.path.join(data_dir, "recorded_speech.wav")
    fs, data = wavfile.read(wav_path)
    signal = data.astype(np.float64) / np.max(np.abs(data))
    duration = len(signal) / fs
    print(f"  Loaded: {wav_path}")
    print(f"  Sample rate: {fs} Hz, Duration: {duration:.2f}s, Samples: {len(signal)}")
    return signal, fs


def plot_spectrogram(signal, fs, nperseg, noverlap, window, ax, title=""):
    f, t_spec, Sxx = spectrogram(signal, fs=fs, window=window,
                                  nperseg=nperseg, noverlap=noverlap)
    ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim([0, fs / 2])
    return f, t_spec, Sxx


def study_window_type(signal, fs, output_dir):
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


def study_nperseg(signal, fs, output_dir):
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


def study_overlap(signal, fs, output_dir):
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


def study_sampling_freq(signal, fs, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fs_values = [8000, 48000, 96000]

    for idx, target_fs in enumerate(fs_values):
        if target_fs == fs:
            resampled = signal
        elif target_fs < fs:
            decimation = fs // target_fs
            resampled = signal[::decimation]
        else:
            interpolation = target_fs // fs
            resampled = np.repeat(signal, interpolation)
        plot_spectrogram(resampled, target_fs, nperseg=1024, noverlap=768,
                         window="hann", ax=axes[idx],
                         title=f"fs={target_fs} Hz\n(max freq={target_fs//2} Hz)")

    plt.suptitle("Effect of Sampling Frequency on Spectrogram", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectrogram_fs_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved spectrogram_fs_comparison.png")


def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, "part2_audio_pitch", "data")
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 2, Step 1: Spectrogram Visualization & Parameter Study ===\n")
    print("Loading recorded audio...")
    signal, fs = load_recorded_audio(data_dir)
    print()

    print("A) Comparing window types...")
    study_window_type(signal, fs, output_dir)

    print("B) Comparing window sizes (nperseg)...")
    study_nperseg(signal, fs, output_dir)

    print("C) Comparing overlap values...")
    study_overlap(signal, fs, output_dir)

    print("D) Comparing sampling frequencies...")
    study_sampling_freq(signal, fs, output_dir)

    print("\nDone! All figures saved to figures/")


if __name__ == "__main__":
    main()
