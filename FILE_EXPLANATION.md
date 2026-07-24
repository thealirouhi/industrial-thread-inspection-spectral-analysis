# Project File-by-File Explanation

Complete inventory of every file in the project, what it does, and why it exists.

---

## Root Level Files

### `FinalProject.pdf`
The project assignment (in Persian). Contains the requirements for both parts of the project — bolt thread detection and audio pitch detection. This is what we're solving.

### `README.md`
Project documentation. Describes the project structure, setup instructions, and what each step does. Also contains the analysis sections (parameter effects, time-frequency trade-offs, Nyquist considerations) required by the assignment.

### `PRESENTATION.md`
Presentation guide. Explains the entire project in simple terms — the concepts, methods, formulas, and results. Structured for a ~20 minute presentation.

### `requirements.txt`
Python dependencies needed to run the project:
- `numpy` — array math, FFT
- `scipy` — convolution, spectrogram, WAV file reading
- `matplotlib` — plotting graphs and images
- `Pillow` — loading and saving images

### `.gitignore`
Tells Git which files to ignore (like `.DS_Store`, `__pycache__`, etc.).

---

## `dataset/` — Real Bolt Images

This is the **main dataset** used by Part 1. Contains real photographs of bolts taken from a factory production line.

### `dataset/threaded/` (10 images)
Real photos of bolts **with threads**. Filenames like `IMG_20250201_154606_296.jpg` are from the actual camera. These are the "positive" samples.

### `dataset/threadless/` (10 images)
Real photos of bolts **without threads** (smooth). These are the "negative" samples — the defective bolts the factory wants to detect.

**Total: 20 real bolt images (10 threaded + 10 threadless)**

---

## `part1_bolt_detection/` — Bolt Thread Detection

### `part1_bolt_detection/step1_convolution.py`
**Purpose:** Detect threads using 2D cosine convolution.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `create_cosine_filter(size, freq, direction)` | Creates a 2D cosine pattern `cos(ax + by)` with a Hanning window to prevent edge artifacts |
| `load_image(path, downsample)` | Loads an image, converts to grayscale, downsamples 4x for speed |
| `detect_threads_convolution(image)` | Tries all combinations of filter sizes, frequencies, and directions. Returns the convolution with highest energy |
| `visualize_filter_and_convolution()` | Creates a 3-panel plot: original image, filter, convolution result |
| `visualize_sample_results()` | Runs detection on sample images and saves figures |
| `main()` | Entry point — runs everything |

**What it produces:**
- `convolution_threaded.png` — shows threaded bolt + best filter + convolution result
- `convolution_threadless.png` — shows unthreaded bolt + best filter + convolution result
- `sample_bolts.png` — side-by-side comparison of threaded vs unthreaded

**The algorithm:**
```
For each filter size (15, 25):
  For each frequency (0.5, 1.0, 1.5):
    For each direction (0°, 45°, 90°, 135°):
      Create cosine filter
      Convolve with image
      Measure energy (mean of squared values)
      Keep the best one
```

**Why cosine filters?** Threads are repeating parallel lines. A cosine function creates exactly this pattern. When the filter aligns with the threads, the convolution response is strongest.

---

### `part1_bolt_detection/step2_fourier_transform.py`
**Purpose:** Detect threads using 2D Fourier Transform.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `compute_2d_fft(image)` | Applies `fft2` + `fftshift` to get centered frequency representation |
| `log_magnitude(mag)` | Applies `log1p` for better visualization (compresses dynamic range) |
| `zero_center_pixels(F, radius)` | Zeros out a circular region at the center (low frequencies = bolt body) |
| `reconstruct_image(F)` | Applies inverse FFT to see what's left after filtering |
| `predictor(image, threshold)` | Returns 1 (threaded) or 0 (unthreaded) based on high-frequency energy |
| `visualize_cosine_fft()` | Shows cosine signals and their FFTs to build intuition |
| `visualize_fft_comparison()` | Compares FFT of threaded vs unthreaded bolts |
| `visualize_predictor_results()` | Runs predictor on all images, plots energy bar chart with threshold |
| `main()` | Entry point — runs all steps A through D |

**What it produces:**
- `cosine_fft_intuition.png` — 5 cosine patterns and their FFT spectra (builds intuition)
- `fft_comparison.png` — side-by-side: threaded vs unthreaded, original, FFT, reconstructed
- `fft_predictor_energy.png` — bar chart showing energy for all images with decision threshold

**The algorithm:**
```
1. Compute FFT of image
2. Zero out center pixels (remove bolt body = low frequencies)
3. Compute energy of remaining frequencies (high frequencies = threads)
4. If energy > threshold → return 1 (threaded)
   Else → return 0 (unthreaded)
```

**Why zero the center?** The bolt body is a large structure occupying low frequencies. Threads are fine details at higher frequencies. Removing the center isolates the thread information.

**Intuition building:** Before applying FFT to bolts, we first show what happens when you take the FFT of simple cosine signals. A cosine at frequency f in direction θ appears as two bright dots in the FFT at (±f, ±θ). This helps understand why threaded bolts (which look like cosines) have different FFT patterns than smooth bolts.

---

### `part1_bolt_detection/step3_noise_analysis.py`
**Purpose:** Test how robust the detector is to noise.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `predictor(image)` | Same FFT-based detector from Step 2 (returns energy value) |
| `compute_threshold(images, labels)` | Finds the optimal threshold from clean images |
| `compute_accuracy(images, labels, n, threshold)` | Adds noise level n, runs predictor, computes accuracy |
| `main()` | Tests accuracy across 18 noise levels, generates plots |

**What it produces:**
- `accuracy_vs_noise.png` — line plot showing accuracy dropping as noise increases
- `noise_examples.png` — visual examples of noisy bolt images at different noise levels

**The noise model:**
```
noisyImage = image + N(0, n)
```
Where `N(0, n)` is Gaussian noise with mean 0 and standard deviation n.

**Noise levels tested:** 0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0

**What the plot shows:**
- Low noise: accuracy is stable (detector works)
- Medium noise: accuracy starts dropping
- High noise: accuracy hits ~50% (random guessing — detector is useless)

---

## `part2_audio_pitch/` — Audio Pitch Detection

### `part2_audio_pitch/data/recorded_speech.wav`
**The recorded audio file.** 9.24 seconds, 48kHz sample rate, mono. This is a sentence recorded by the user. Used by Step 1 and Step 3.

---

### `part2_audio_pitch/step1_spectrogram.py`
**Purpose:** Visualize spectrograms and study how parameters affect them.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `load_recorded_audio(data_dir)` | Loads `recorded_speech.wav`, converts to float64 normalized to [-1, 1] |
| `plot_spectrogram(signal, fs, nperseg, noverlap, window, ax)` | Computes and plots one spectrogram using `scipy.signal.spectrogram` |
| `study_window_type(signal, fs, output_dir)` | Compares 6 window types: hann, hamming, blackman, boxcar, flattop, tukey(0.5) |
| `study_nperseg(signal, fs, output_dir)` | Compares window sizes: 256, 1024, 4096 |
| `study_overlap(signal, fs, output_dir)` | Compares overlap: 0, 256, 768 samples |
| `study_sampling_freq(signal, fs, output_dir)` | Compares sampling rates: 8000, 48000, 96000 Hz |
| `main()` | Loads recorded audio, runs all 4 studies |

**What it produces:**
- `spectrogram_window_comparison.png` — 2×3 grid showing different windows
- `spectrogram_nperseg_comparison.png` — 3 spectrograms with different window sizes
- `spectrogram_overlap_comparison.png` — 3 spectrograms with different overlaps
- `spectrogram_fs_comparison.png` — 3 spectrograms with different sampling rates

**What each parameter does:**

| Parameter | Effect |
|-----------|--------|
| Window type | Controls how spectral energy leaks to adjacent frequencies |
| nperseg | Larger = better frequency resolution, worse time resolution |
| Overlap | More overlap = smoother spectrogram, more frames, slower |
| Sampling freq | Higher = wider observable frequency range (up to fs/2) |

---

### `part2_audio_pitch/step2_synthetic_signals.py`
**Purpose:** Compare spectrogram vs FFT for different waveform types.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `generate_sinusoidal_signals()` | Creates 3 concatenated sine waves at 100Hz, 150Hz, 180Hz |
| `generate_square_signals()` | Creates 3 concatenated square waves at same frequencies |
| `generate_triangle_signals()` | Creates 3 concatenated triangle waves at same frequencies |
| `plot_spectrogram_and_fft()` | Plots spectrogram and FFT side-by-side for comparison |
| `main()` | Generates all 3 wave types, creates combined figure |

**What it produces:**
- `synthetic_spectrogram_vs_fft.png` — 3×2 grid: each row is a waveform type, left column is spectrogram, right column is FFT

**Key observations:**
- **Sine:** Clean single lines in spectrogram; clean peaks in FFT
- **Square:** Multiple harmonic lines (odd harmonics); more peaks in FFT
- **Triangle:** Harmonic lines decaying faster; cleaner than square
- **Spectrogram shows time:** You can see when each frequency starts/stops
- **FFT shows only frequency:** No timing information, just overall content

**Signal parameters:**
- Duration per segment: 1 second (48000 samples each)
- Frequencies: 100Hz, 150Hz, 180Hz
- Sampling rate: 48000 Hz (well above Nyquist for 180Hz)
- Periods per segment: 100 (more than enough for clean FFT)

---

### `part2_audio_pitch/step3_fundamental_freq.py`
**Purpose:** Extract the fundamental frequency f0(t) from recorded speech.

**Key functions:**

| Function | What it does |
|----------|-------------|
| `load_recorded_audio(data_dir)` | Loads the recorded WAV file |
| `extract_fundamental_freq(signal, fs, nperseg, noverlap)` | For each spectrogram column, finds the peak frequency (>50Hz) as f0 |
| `main()` | Loads audio, extracts f0(t), plots spectrogram + f0 overlay, prints trade-off analysis |

**What it produces:**
- `fundamental_frequency_extraction.png` — 2 panels: spectrogram on top, f0(t) line plot on bottom
- Printed analysis of time-frequency resolution trade-off

**The algorithm:**
```
For each time frame (column in spectrogram):
    1. Look at the spectrum (one column of Sxx)
    2. Ignore frequencies below 50Hz (noise floor)
    3. Find the frequency with maximum energy
    4. That's f0 at this moment
```

**What the plot shows:**
- Top panel: spectrogram of the recorded speech
- Bottom panel: blue line showing how f0 (pitch) changes over time
- You can see when the speaker's voice goes up or down in pitch

**Printed trade-off analysis:**
```
Frequency resolution: Δf = fs/nperseg = 48000/1024 = 46.9 Hz
Time resolution: Δt = (nperseg-noverlap)/fs = (1024-768)/48000 = 5.3 ms
Max frequency: fs/2 = 24000 Hz (Nyquist limit)
```

---

## `figures/` — Generated Output

All plots and visualizations produced by running the scripts. 15 figures total:

| File | Produced by | What it shows |
|------|------------|---------------|
| `sample_bolts.png` | step1_convolution.py | Side-by-side threaded vs unthreaded bolt photos |
| `convolution_threaded.png` | step1_convolution.py | Threaded bolt + best cosine filter + convolution result |
| `convolution_threadless.png` | step1_convolution.py | Unthreaded bolt + best cosine filter + convolution result |
| `cosine_fft_intuition.png` | step2_fourier_transform.py | 5 cosine signals and their FFT spectra |
| `fft_comparison.png` | step2_fourier_transform.py | Threaded vs unthreaded: original, FFT, reconstructed |
| `fft_predictor_energy.png` | step2_fourier_transform.py | Bar chart of energy values with decision threshold |
| `accuracy_vs_noise.png` | step3_noise_analysis.py | Detection accuracy vs noise level curve |
| `noise_examples.png` | step3_noise_analysis.py | Visual examples of noisy bolt images |
| `spectrogram_window_comparison.png` | step1_spectrogram.py | Effect of window type on spectrogram |
| `spectrogram_nperseg_comparison.png` | step1_spectrogram.py | Effect of window size on spectrogram |
| `spectrogram_overlap_comparison.png` | step1_spectrogram.py | Effect of overlap on spectrogram |
| `spectrogram_fs_comparison.png` | step1_spectrogram.py | Effect of sampling frequency on spectrogram |
| `synthetic_spectrogram_vs_fft.png` | step2_synthetic_signals.py | Sine/square/triangle: spectrogram vs FFT |
| `fundamental_frequency_extraction.png` | step3_fundamental_freq.py | Spectrogram + extracted f0(t) from recorded speech |

---

## How to Run Everything

```bash
# Install dependencies
pip install -r requirements.txt

# Part 1: Bolt Detection
python3 part1_bolt_detection/step1_convolution.py
python3 part1_bolt_detection/step2_fourier_transform.py
python3 part1_bolt_detection/step3_noise_analysis.py

# Part 2: Audio Pitch
python3 part2_audio_pitch/step1_spectrogram.py
python3 part2_audio_pitch/step2_synthetic_signals.py
python3 part2_audio_pitch/step3_fundamental_freq.py
```

All scripts output their figures to the `figures/` directory.

---

## File Count Summary

| Category | Count |
|----------|-------|
| Python scripts | 6 |
| Real bolt images | 20 (10 threaded + 10 threadless) |
| Audio files | 1 (recorded_speech.wav) |
| Generated figures | 15 |
| Documentation | 3 (README.md, PRESENTATION.md, this file) |
| Config | 2 (requirements.txt, .gitignore) |
| **Total project files** | **~50** |
