import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_image(path, downsample=4):
    img = Image.open(path).convert("L")
    w, h = img.size
    img = img.resize((w // downsample, h // downsample))
    return np.array(img, dtype=np.float64) / 255.0


def compute_2d_fft(image):
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)
    magnitude = np.abs(F_shifted)
    return F_shifted, magnitude


def log_magnitude(mag):
    return np.log1p(mag)


def zero_center_pixels(F_shifted, radius=5):
    F_filtered = F_shifted.copy()
    h, w = F_filtered.shape
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[:h, :w]
    mask = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    F_filtered[mask] = 0
    return F_filtered


def reconstruct_image(F_filtered):
    F_ishift = np.fft.ifftshift(F_filtered)
    img_recon = np.fft.ifft2(F_ishift)
    return np.abs(img_recon)


def visualize_fft_comparison(t_img, u_img, output_dir, radius=5):
    F_t, mag_t = compute_2d_fft(t_img)
    F_u, mag_u = compute_2d_fft(u_img)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(t_img, cmap="gray")
    axes[0, 0].set_title("Threaded Bolt")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(log_magnitude(mag_t), cmap="hot")
    axes[0, 1].set_title("FFT Magnitude (Log) - Threaded")
    axes[0, 1].axis("off")

    F_t_filtered = zero_center_pixels(F_t, radius=radius)
    recon_t = reconstruct_image(F_t_filtered)
    axes[0, 2].imshow(recon_t, cmap="gray")
    axes[0, 2].set_title(f"Reconstructed (center zeroed, r={radius})")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(u_img, cmap="gray")
    axes[1, 0].set_title("Unthreaded Bolt")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(log_magnitude(mag_u), cmap="hot")
    axes[1, 1].set_title("FFT Magnitude (Log) - Unthreaded")
    axes[1, 1].axis("off")

    F_u_filtered = zero_center_pixels(F_u, radius=radius)
    recon_u = reconstruct_image(F_u_filtered)
    axes[1, 2].imshow(recon_u, cmap="gray")
    axes[1, 2].set_title(f"Reconstructed (center zeroed, r={radius})")
    axes[1, 2].axis("off")

    plt.suptitle("2D Fourier Transform: Threaded vs Unthreaded Bolt", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fft_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fft_comparison.png")


def visualize_cosine_fft(output_dir):
    freqs_and_dirs = [(0.5, 0), (0.5, 45), (1.0, 0), (1.0, 45), (1.5, 90)]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    size = 128
    Y, X = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2
    x = X - cx
    y = Y - cy

    for col, (freq, direction) in enumerate(freqs_and_dirs):
        theta = np.radians(direction)
        a = freq * np.cos(theta)
        b = freq * np.sin(theta)
        cosine_2d = np.cos(a * x + b * y)

        F, mag = compute_2d_fft(cosine_2d)

        axes[0, col].imshow(cosine_2d, cmap="RdBu_r")
        axes[0, col].set_title(f"cos({a:.1f}x+{b:.1f}y)\nf={freq}, dir={direction}")
        axes[0, col].axis("off")

        axes[1, col].imshow(log_magnitude(mag), cmap="hot")
        axes[1, col].set_title("FFT Magnitude (Log)")
        axes[1, col].axis("off")

    plt.suptitle("2D Cosine Signals and Their Fourier Transforms", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cosine_fft_intuition.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cosine_fft_intuition.png")


def predictor(image, radius=5):
    F, mag = compute_2d_fft(image)
    F_filtered = zero_center_pixels(F, radius=radius)
    energy = np.mean(np.abs(F_filtered)**2)
    return energy


def get_image_files(data_dir, cat):
    cat_dir = os.path.join(data_dir, cat)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(exts)])


def visualize_predictor_results(data_dir, output_dir, radius=5):
    categories = ["threaded", "threadless"]
    energies_threaded = []
    energies_unthreaded = []

    for cat in categories:
        files = get_image_files(data_dir, cat)
        for f in files:
            img = load_image(os.path.join(data_dir, cat, f))
            energy = predictor(img, radius=radius)
            if cat == "threaded":
                energies_threaded.append(energy)
            else:
                energies_unthreaded.append(energy)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(energies_threaded)), energies_threaded, alpha=0.7, label="Threaded", color="steelblue")
    ax.bar(range(len(energies_unthreaded)), energies_unthreaded, alpha=0.7, label="Unthreaded", color="salmon")
    threshold = (np.mean(energies_threaded) + np.mean(energies_unthreaded)) / 2
    ax.axhline(y=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.1f})")
    ax.set_xlabel("Image Index")
    ax.set_ylabel("High-Frequency Energy")
    ax.set_title("FFT-Based Thread Detection: High-Frequency Energy Comparison")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fft_predictor_energy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fft_predictor_energy.png")

    correct = 0
    total = len(energies_threaded) + len(energies_unthreaded)
    for e in energies_threaded:
        if e > threshold:
            correct += 1
    for e in energies_unthreaded:
        if e <= threshold:
            correct += 1

    accuracy = correct / total * 100
    print(f"\n  Predictor accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"  Threaded energy range: [{min(energies_threaded):.2f}, {max(energies_threaded):.2f}]")
    print(f"  Unthreaded energy range: [{min(energies_unthreaded):.2f}, {max(energies_unthreaded):.2f}]")
    return threshold


def main():
    base_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, "dataset")
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 1, Step 2: 2D Fourier Transform for Thread Detection ===\n")

    print("Step A: Building intuition with cosine signals...")
    visualize_cosine_fft(output_dir)

    print("\nStep B: Comparing threaded vs unthreaded FFT spectra...")
    t_files = get_image_files(data_dir, "threaded")
    u_files = get_image_files(data_dir, "threadless")
    t_img = load_image(os.path.join(data_dir, "threaded", t_files[0]))
    u_img = load_image(os.path.join(data_dir, "threadless", u_files[0]))
    visualize_fft_comparison(t_img, u_img, output_dir, radius=5)

    print("\nStep C: Running predictor on all images...")
    threshold = visualize_predictor_results(data_dir, output_dir, radius=5)

    print(f"\nDone! Figures saved to figures/")
    return threshold


if __name__ == "__main__":
    main()
