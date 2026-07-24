import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
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
    return t_spec, f0_per_frame, f, Sxx


def main():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(project_dir, "part2_audio_pitch", "data")
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 2, Step 3: Fundamental Frequency f0(t) Extraction ===\n")
    print("Loading recorded audio...")
    signal, fs = load_recorded_audio(data_dir)
    print()

    nperseg = 1024
    noverlap = 768
    t_spec, f0, f, Sxx = extract_fundamental_freq(signal, fs, nperseg, noverlap)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax_spec = axes[0]
    ax_spec.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Recorded Speech - Spectrogram")
    ax_spec.set_ylim([0, fs / 2])

    ax_f0 = axes[1]
    ax_f0.plot(t_spec, f0, "b-", linewidth=2, label="Extracted f0(t)")
    ax_f0.set_ylabel("Frequency (Hz)")
    ax_f0.set_xlabel("Time (s)")
    ax_f0.set_title("Recorded Speech - Fundamental Frequency f0(t)")
    ax_f0.legend()
    ax_f0.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fundamental_frequency_extraction.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fundamental_frequency_extraction.png")

    print("\n--- Time-Frequency Resolution Trade-off Analysis ---\n")
    freq_res = fs / nperseg
    time_res = (nperseg - noverlap) / fs
    max_freq = fs / 2
    print(f"  Parameters used: nperseg={nperseg}, noverlap={noverlap}, fs={fs} Hz")
    print(f"  Frequency resolution: delta_f = fs/nperseg = {fs}/{nperseg} = {freq_res:.1f} Hz")
    print(f"  Time resolution: delta_t = (nperseg-noverlap)/fs = ({nperseg}-{noverlap})/{fs} = {time_res*1000:.1f} ms")
    print(f"  Max observable frequency (Nyquist): fs/2 = {max_freq} Hz")
    print()
    print("  Trade-off: Increasing nperseg improves frequency resolution (smaller delta_f)")
    print("  but degrades time resolution (larger delta_t). This is a fundamental")
    print("  limitation similar to the Heisenberg uncertainty principle.")
    print()
    print("  Increasing overlap (noverlap) recovers some time resolution by computing")
    print("  more STFT frames, but increases computation cost proportionally.")
    print()
    print("  For pitch detection, sufficient frequency resolution is needed to")
    print("  distinguish the fundamental frequency f0 from its harmonics (2*f0, 3*f0, ...).")
    print("  Too coarse frequency resolution may cause harmonics to be mistaken for f0.")

    print("\nDone! Figure saved to figures/")


if __name__ == "__main__":
    main()
