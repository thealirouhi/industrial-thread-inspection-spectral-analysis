# Project Presentation Guide

## Industrial Thread Inspection & Spectral Analysis

Signals and Systems Course — Dr. Mirzaei, University of Tehran

---

## What Is This Project About?

This project solves **two real-world problems** using signal processing techniques you learned in class:

1. **Part 1:** Detecting whether bolts on a factory production line have threads or not (using image processing)
2. **Part 2:** Detecting the pitch (fundamental frequency) of a recorded voice over time (using audio processing)

Both parts use the **same core idea**: converting signals into the frequency domain using **Fourier Transform** to reveal information that's hidden in the original signal.

---

## Part 1: Bolt Thread Detection

### The Problem

Imagine a factory that produces bolts. Some bolts are mistakenly **not threaded** (they're smooth). The factory needs to automatically detect and remove these defective bolts from the production line.

**Challenges:**
- Bolts can be at **any angle** on the conveyor belt
- The **distance** from the camera varies
- The **background** may change
- We need a fast, automatic solution

### Our Approach: Two Methods

---

### Method 1: 2D Cosine Convolution

**What is convolution?**
Convolution is like sliding a filter (small matrix) over the image. At each position, we multiply the filter values with the image pixels underneath and sum them up. If the pattern in the image matches the filter pattern, the result is large.

**Why cosine filters?**
Threads on a bolt look like repeating parallel lines at a certain angle. A cosine function `cos(ax + by)` creates exactly this kind of pattern — repeating stripes at a specific angle and frequency.

**How it works:**
```
1. Load the bolt image (convert to grayscale, downsample 4x)
2. Create cosine filters with different:
   - Frequencies (how close the threads are)
   - Directions (threads can be at any angle)
3. Convolve each filter with the image
4. The filter that gives the HIGHEST energy = best match
5. High energy = threads present, Low energy = no threads
```

**Why downsample?**
Convolution is slow on large images. Downsampling by 4x makes it 16x faster. This is safe because thread patterns are relatively large features (many pixels), so reducing resolution doesn't lose the thread information. This respects the **Nyquist theorem** — we're not losing the frequencies we care about.

**Key公式 (Formula):**
```
cos(ax + by)
where:
  a = frequency × cos(direction)
  b = frequency × sin(direction)
```

---

### Method 2: 2D Fourier Transform

**What is Fourier Transform?**
Any signal (or image) can be broken down into a sum of sine/cosine waves at different frequencies. The Fourier Transform tells us **which frequencies are present** and **how strong each one is**.

**Why does this help?**
- A **smooth bolt** (no threads) = mostly low frequencies (the big round shape)
- A **threaded bolt** = has extra high frequencies (the fine thread patterns)

By looking at the frequency spectrum, we can tell them apart!

**How it works:**
```
1. Load the bolt image
2. Apply 2D FFT (fft2) → get frequency representation
3. Use fftshift to center the zero-frequency component
4. Take the log of magnitude for better visualization
5. ZERO OUT the center pixels (low frequencies = bolt body)
6. Measure the remaining energy (high frequencies = threads)
7. If energy > threshold → threaded (return 1)
   If energy ≤ threshold → unthreaded (return 0)
```

**Why zero out the center?**
The bolt body is a large structure that dominates low frequencies. By removing it, we isolate the thread patterns which live in higher frequencies. It's like turning down the bass to hear the treble clearly.

**The predictor function:**
```python
def predictor(image, threshold):
    F = fft2(image)                    # Convert to frequency domain
    F_shifted = fftshift(F)            # Center zero frequency
    F_filtered = zero_center(F_shifted) # Remove low frequencies
    energy = mean(|F_filtered|²)       # Measure high-freq energy
    if energy > threshold:
        return 1  # threaded
    else:
        return 0  # unthreaded
```

---

### Step 3: Noise Robustness Test

**What we tested:**
How well does our detector work when the image gets noisy (like a bad camera or poor lighting)?

**How:**
```
noisyImage = originalImage + N(0, n)
```
Where `N(0, n)` is Gaussian noise with mean 0 and standard deviation `n`.

We tested noise levels from `n = 0` (clean) to `n = 100,000` (completely destroyed image) and plotted accuracy vs noise level.

**Result:**
- At low noise: accuracy is high
- As noise increases: accuracy drops
- At very high noise: accuracy approaches 50% (random guessing, since we have equal numbers of threaded/unthreaded bolts)

This tells us the **operating range** of our detector — how much noise it can handle before becoming useless.

---

## Part 2: Audio Pitch Detection

### The Problem

We want to track how the **pitch** (fundamental frequency f0) of a voice changes over time. This is useful for:
- Speech recognition
- Music analysis
- Speaker identification

**Why not just use regular Fourier Transform?**
Regular FFT assumes the signal doesn't change over time. But speech IS changing — you say different sounds at different times. We need to know **when** each frequency occurs, not just **which** frequencies exist.

**Solution: Short-Time Fourier Transform (STFT)**

---

### Step 1: Spectrogram Visualization

**What is a spectrogram?**
A spectrogram is like a movie of the Fourier Transform. Instead of one FFT for the whole signal, we:
```
1. Take a short window of the signal (e.g., 1024 samples)
2. Compute FFT of that window
3. Move the window forward (with some overlap)
4. Repeat
5. Stack all the FFTs side by side → creates an image
```

**The result:** An image where:
- **X-axis** = time
- **Y-axis** = frequency
- **Color** = how strong each frequency is at each moment

**Parameter Studies:**

| Parameter | What it does | Trade-off |
|-----------|-------------|-----------|
| **Window type** (hann, hamming, blackman, boxcar) | Affects spectral leakage | Hann/Blackman: better freq resolution but wider main lobe. Boxcar: narrow main lobe but high sidelobes |
| **nperseg** (window size: 256, 1024, 4096) | Controls time vs frequency resolution | Larger → better freq resolution, worse time resolution. Smaller → opposite |
| **Overlap** (0%, 25%, 75%) | How much windows overlap | More overlap → smoother spectrogram, better time resolution, but slower |
| **Sampling freq** (8000, 48000, 96000 Hz) | Max observable frequency | Higher fs → can see higher frequencies, but doesn't improve freq resolution |

---

### Step 2: Synthetic Signal Analysis

**What we did:**
Created three types of signals with known frequencies (100Hz, 150Hz, 180Hz) and compared their spectrograms:

**1. Sinusoidal signals:**
- Pure tones → show as clean horizontal lines in spectrogram
- FFT shows peaks at exactly 100, 150, 180 Hz
- Spectrogram shows WHEN each frequency is active

**2. Square waves:**
- Contain odd harmonics (f0, 3f0, 5f0, 7f0, ...)
- Spectrogram shows all harmonics simultaneously
- More complex spectrum than pure sine

**3. Triangle waves:**
- Also odd harmonics but decaying faster (1/n² vs 1/n)
- Cleaner spectrum than square wave
- Still richer than pure sine

**Key observation:** Spectrogram shows time information (when each frequency appears/disappears), while FFT only shows overall frequency content.

---

### Step 3: Fundamental Frequency Extraction

**What is f0?**
The fundamental frequency is the lowest frequency component of a harmonic signal. For speech, it's the "pitch" you hear — high f0 = high pitch, low f0 = low pitch.

**How we extract it:**
```
For each column (time frame) in the spectrogram:
    1. Find the frequency with the HIGHEST energy (peak)
    2. That peak is f0 at that moment
    3. Skip frequencies below 50Hz (usually noise)
```

**Time-Frequency Resolution Trade-off:**

This is a fundamental limitation (like Heisenberg uncertainty principle):

```
Δf = fs / nperseg    (frequency resolution)
Δt = (nperseg - noverlap) / fs  (time resolution)
```

- You **cannot** have both high frequency resolution AND high time resolution
- Larger window → better frequency precision, worse time precision
- Overlap helps recover some time resolution but costs more computation

For pitch detection, you need enough frequency resolution to distinguish f0 from its harmonics (2f0, 3f0, etc.).

---

## Summary of Techniques Used

| Technique | Where Used | Purpose |
|-----------|-----------|---------|
| 2D Convolution | Part 1, Step 1 | Match thread patterns with cosine filters |
| 2D FFT (fft2, fftshift) | Part 1, Step 2 | Analyze frequency content of bolt images |
| Gaussian Noise Addition | Part 1, Step 3 | Test detector robustness |
| STFT Spectrogram | Part 2, Steps 1-3 | Time-frequency analysis of audio |
| Peak Detection | Part 2, Step 3 | Extract fundamental frequency f0(t) |

---

## Key Results to Show in Presentation

**Part 1:**
- Convolution results showing thread detection at different angles
- FFT comparison: threaded vs unthreaded bolt spectra
- Predictor accuracy plot (energy distribution with threshold)
- Noise robustness curve (accuracy vs noise level)

**Part 2:**
- Spectrogram parameter comparison (window type, nperseg, overlap, fs)
- Synthetic signal spectrograms vs FFT (showing time localization)
- Fundamental frequency f0(t) extraction from recorded speech

---

## Presentation Flow (Suggested)

1. **Introduction** (2 min): Problem statement for both parts
2. **Part 1 - Convolution** (3 min): Show cosine filters, convolution results
3. **Part 1 - Fourier** (3 min): Show FFT spectra, explain center zeroing, show predictor
4. **Part 1 - Noise** (2 min): Show accuracy curve, discuss robustness
5. **Part 2 - Spectrogram** (3 min): Explain STFT, show parameter effects
6. **Part 2 - Synthetic** (2 min): Compare spectrogram vs FFT
7. **Part 2 - f0 Extraction** (3 min): Show pitch tracking, discuss trade-offs
8. **Conclusion** (2 min): Summary of techniques and results

Total: ~20 minutes

---

## Key Terms to Know

- **Convolution**: Sliding a filter over a signal/image to detect patterns
- **Fourier Transform**: Decomposing a signal into frequency components
- **FFT (fft2)**: Fast algorithm for computing Fourier Transform
- **fftshift**: Moving zero-frequency to the center for visualization
- **STFT**: Short-Time Fourier Transform — FFT applied to sliding windows
- **Spectrogram**: Visual representation of STFT (time vs frequency vs amplitude)
- **f0 (fundamental frequency)**: The lowest frequency component (pitch)
- **Nyquist theorem**: Sampling rate must be ≥ 2× the highest frequency
- **Spectral leakage**: Energy "leaking" to adjacent frequencies due to windowing
- **Window function**: Tapering function applied before FFT to reduce spectral leakage
