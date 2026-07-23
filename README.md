# Industrial Thread Inspection & Spectral Analysis

Signals and Systems final project implementing two core signal processing applications:

1. **Bolt Thread Detection** -- Identifying threaded vs non-threaded bolts on a production line using 2D convolution and Fourier Transform
2. **Audio Pitch Detection** -- Extracting fundamental frequency f0(t) over time using STFT spectrogram analysis

## Project Structure

```
.
├── part1_bolt_detection/
│   ├── data/                    # Threaded and unthreaded bolt images
│   ├── step1_convolution.py     # 2D cosine convolution for thread detection
│   ├── step2_fourier_transform.py # 2D FFT-based thread detection
│   └── step3_noise_analysis.py  # Accuracy vs Gaussian noise level
├── part2_audio_pitch/
│   ├── data/                    # Recorded audio files
│   ├── step1_spectrogram.py     # STFT spectrogram visualization & parameter study
│   ├── step2_synthetic_signals.py # Spectrograms of synthetic signals
│   └── step3_fundamental_freq.py # f0(t) extraction from spectrogram
├── figures/                     # Generated plots and results
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Part 1: Bolt Thread Detection

### Step 1 -- Convolution Approach
- Load bolt images and convert to single channel (grayscale)
- Design 2D cosine filters: cos(ax + by) with various frequencies and directions
- Convolve filters with images to highlight thread patterns
- Apply thresholding for automatic detection

**Nyquist Consideration:** Images are downsampled by 4x before processing. This respects the Nyquist sampling theorem because bolt thread patterns are relatively low-frequency features compared to the original pixel resolution. The thread spacing spans many pixels, so a 4x reduction still preserves sufficient detail for detection while significantly speeding up convolution.

### Step 2 -- Fourier Transform Approach
- Apply 2D FFT (fft2) and shift zero-frequency to center (fftshift)
- Compare frequency spectra of threaded vs unthreaded bolts
- Zero out low-frequency center pixels (bolt body) to enhance thread detection
- Build automatic `predictor()` function: returns 1 (threaded) or 0 (unthreaded)

**Key Insight:** The bolt body is a large structure occupying low frequencies (center of FFT), while thread patterns are fine details occupying higher frequencies. By zeroing the center pixels and measuring remaining high-frequency energy, we can distinguish threaded from unthreaded bolts. The `predictor()` function uses a threshold on this energy to make a binary classification (1 = threaded, 0 = unthreaded).

### Step 3 -- Noise Robustness
- Add Gaussian noise: `noisyImage = image + N(0, n)`
- Evaluate accuracy across noise levels (n = 0 to n = 100,000)
- Plot accuracy vs noise level curve

## Part 2: Audio Pitch Detection

### Step 1 -- Spectrogram Visualization
- Load recorded audio (`recorded_speech.wav`) and compute STFT spectrogram using `scipy.signal.spectrogram`
- Experiment with parameters: overlap, window type, nperseg (window size), sampling frequency

#### Parameter Analysis

| Parameter | Values Tested | Effect on Spectrogram |
|-----------|---------------|----------------------|
| **Window type** | hann, hamming, blackman, boxcar, flattop, tukey(0.5) | Controls spectral leakage and resolution. Hann/Blackman provide better frequency resolution but wider main lobes. Boxcar has the narrowest main lobe but high sidelobes causing spectral leakage. Flattop has flat passband for amplitude accuracy. |
| **nperseg (window size)** | 256, 1024, 4096 | Controls the time-frequency resolution trade-off. Larger nperseg gives finer frequency resolution (delta_f = fs/nperseg) but coarser time resolution. Smaller nperseg gives better time localization but broader frequency peaks. |
| **Overlap** | 0%, 25%, 75% | Higher overlap produces smoother spectrograms with better time resolution by computing more STFT frames. However, it increases computation time proportionally. |
| **Sampling frequency** | 8000, 48000, 96000 Hz | Determines the maximum observable frequency (Nyquist = fs/2). Higher fs allows observing higher frequency components but does not improve frequency resolution (which depends on nperseg). |

### Step 2 -- Synthetic Signal Analysis
- Generate sinusoidal signals at 100Hz, 150Hz, 180Hz (concatenated)
- Generate square and triangle wave equivalents
- Compare spectrogram vs FFT visualization

#### Observations
- **Sinusoidal signals:** Show clean single-frequency lines in both spectrogram and FFT. The spectrogram reveals when each frequency is active (time-localized), while the FFT shows only the overall frequency content without timing information.
- **Square waves:** Display odd harmonics (f0, 3*f0, 5*f0, ...) with decaying amplitudes. The spectrogram shows these harmonics appearing and disappearing at segment boundaries.
- **Triangle waves:** Also show odd harmonics but with faster decay (1/n^2 vs 1/n for square waves). The spectrogram clearly shows frequency transitions between segments.

### Step 3 -- Fundamental Frequency Extraction
- Extract f0(t) from spectrogram at each time column (each STFT frame)
- Plot fundamental frequency evolution over time

#### Time-Frequency Resolution Trade-off

The fundamental limitation of STFT is the time-frequency uncertainty principle:

- **Frequency resolution:** delta_f = fs / nperseg (e.g., 48000/1024 = 46.9 Hz)
- **Time resolution:** delta_t = (nperseg - noverlap) / fs (e.g., (1024-768)/48000 = 5.3 ms)

Increasing nperseg improves frequency resolution (smaller delta_f) but degrades time resolution (larger delta_t). This is analogous to the Heisenberg uncertainty principle -- we cannot simultaneously achieve arbitrarily high resolution in both domains.

Increasing overlap recovers some time resolution by computing more STFT frames, but increases computation cost proportionally. For pitch detection, sufficient frequency resolution is needed to distinguish f0 from its harmonics (2*f0, 3*f0, ...). Too coarse frequency resolution may cause harmonics to be mistaken for the fundamental frequency.

## Tools Used

- **Python** with NumPy, SciPy, Matplotlib, Pillow
- 2D Fourier Transform (fft2, fftshift)
- Short-Time Fourier Transform (STFT spectrogram)
- 2D convolution with directional cosine filters

## References

- Signals and Systems course, Dr. Mirzaei, University of Tehran
- [Related paper on bolt thread detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5615672)
