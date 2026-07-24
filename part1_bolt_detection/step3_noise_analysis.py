import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def load_image(path, target_size=256):
    img = Image.open(path).convert("L")
    img = img.resize((target_size, target_size))
    return np.array(img, dtype=np.float64) / 255.0


def compute_2d_fft(image):
    return np.fft.fftshift(np.fft.fft2(image))


def zero_center_pixels(F_shifted, radius=5):
    F_filtered = F_shifted.copy()
    h, w = F_filtered.shape
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[:h, :w]
    mask = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    F_filtered[mask] = 0
    return F_filtered


def predictor(image, radius=5):
    F = compute_2d_fft(image)
    F_filtered = zero_center_pixels(F, radius=radius)
    return np.mean(np.abs(F_filtered)**2)


def get_image_files(data_dir, cat):
    cat_dir = os.path.join(data_dir, cat)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(exts)])


def compute_threshold(images, labels, radius=5):
    energies_threaded = []
    energies_unthreaded = []
    for img, label in zip(images, labels):
        energy = predictor(img, radius=radius)
        if label == 1:
            energies_threaded.append(energy)
        else:
            energies_unthreaded.append(energy)
    return (np.mean(energies_threaded) + np.mean(energies_unthreaded)) / 2


def compute_accuracy(images, labels, n, threshold, radius=5):
    correct = 0
    for img, label in zip(images, labels):
        noisy_img = img + np.random.normal(0, n, img.shape)
        noisy_img = np.clip(noisy_img, 0, 1)
        energy = predictor(noisy_img, radius=radius)
        predicted = 1 if energy > threshold else 0
        if predicted == label:
            correct += 1
    return correct / len(images)


def main():
    base_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, "dataset")
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 1, Step 3: Noise Robustness Analysis ===\n")
    print("  All images resized to 256x256 (fixed size for comparable FFT energy)\n")

    all_images = []
    all_labels = []
    for f in get_image_files(data_dir, "threaded"):
        all_images.append(load_image(os.path.join(data_dir, "threaded", f)))
        all_labels.append(1)
    for f in get_image_files(data_dir, "threadless"):
        all_images.append(load_image(os.path.join(data_dir, "threadless", f)))
        all_labels.append(0)

    threshold = compute_threshold(all_images, all_labels)
    print(f"Computed threshold from clean images: {threshold:.2f}\n")

    n_values = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0, 1000000.0]
    accuracies = []

    print("Testing accuracy at various noise levels...")
    for n in n_values:
        np.random.seed(42)
        acc = compute_accuracy(all_images, all_labels, n, threshold)
        accuracies.append(acc * 100)
        print(f"  n = {n:8.2f} -> accuracy = {acc*100:5.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_values, accuracies, "bo-", linewidth=2, markersize=6)
    ax.set_xlabel("Noise Standard Deviation (n)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Detection Accuracy vs Gaussian Noise Level", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color="r", linestyle="--", alpha=0.5, label="Random Chance (50%)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_vs_noise.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved accuracy_vs_noise.png")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    sample_indices = [0, len(get_image_files(data_dir, "threaded"))]
    sample_images = [all_images[i] for i in sample_indices]
    sample_labels = ["Threaded", "Unthreaded"]
    noise_examples = [0, 0.1, 1.0, 5.0, 50.0]

    for row, (img, label) in enumerate(zip(sample_images, sample_labels)):
        for col, n in enumerate(noise_examples):
            np.random.seed(42)
            noisy = img + np.random.normal(0, n, img.shape)
            noisy = np.clip(noisy, 0, 1)
            axes[row, col].imshow(noisy, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{label}\nn={n}")
            axes[row, col].axis("off")

    plt.suptitle("Noisy Bolt Images at Different Noise Levels", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "noise_examples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved noise_examples.png")

    print("\nDone! Figures saved to figures/")


if __name__ == "__main__":
    main()
