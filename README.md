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
- Load bolt images and convert to single channel
- Design 2D cosine filters: cos(ax + by) with various frequencies and directions
- Convolve filters with images to highlight thread patterns
- Apply thresholding for automatic detection

### Step 2 -- Fourier Transform Approach
- Apply 2D FFT (fft2) and shift zero-frequency to center (fftshift)
- Compare frequency spectra of threaded vs unthreaded bolts
- Zero out low-frequency center pixels (bolt body) to enhance thread detection
- Build automatic `predictor()` function: returns 1 (threaded) or 0 (unthreaded)

### Step 3 -- Noise Robustness
- Add Gaussian noise: `noisyImage = image + N(0, n)`
- Evaluate accuracy across noise levels (n = 0 to n = 1,000,000)
- Plot accuracy vs noise level curve

## Part 2: Audio Pitch Detection

### Step 1 -- Spectrogram Visualization
- Load recorded audio and compute STFT spectrogram using `scipy.signal.spectrogram`
- Experiment with parameters: overlap, window type, sampling frequency
- Analyze effect of each parameter on time-frequency resolution

### Step 2 -- Synthetic Signal Analysis
- Generate sinusoidal signals at 100Hz, 150Hz, 180Hz (concatenated)
- Generate square and triangle wave equivalents
- Compare spectrogram vs FFT visualization

### Step 3 -- Fundamental Frequency Extraction
- Extract f0(t) from spectrogram at each time column
- Plot fundamental frequency evolution over time
- Discuss trade-offs between time and frequency resolution

## Tools Used

- **Python** with NumPy, SciPy, Matplotlib, Pillow
- 2D Fourier Transform (fft2, fftshift)
- Short-Time Fourier Transform (STFT spectrogram)
- 2D convolution with directional cosine filters

## References

- Signals and Systems course, Dr. Mirzaei, University of Tehran
- [Related paper on bolt thread detection](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5615672)
