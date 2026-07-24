import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from PIL import Image
import os


def create_cosine_filter(size, freq, direction_deg):
    theta = np.radians(direction_deg)
    a = freq * np.cos(theta)
    b = freq * np.sin(theta)
    Y, X = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 2
    x = X - cx
    y = Y - cy
    filt = np.cos(a * x + b * y)
    window = np.hanning(size)[:, None] * np.hanning(size)[None, :]
    filt = filt * window
    return filt


def load_image(path, target_size=256):
    img = Image.open(path).convert("L")
    img = img.resize((target_size, target_size))
    return np.array(img, dtype=np.float64) / 255.0


def detect_threads_convolution(image, filter_sizes=[15, 25], freqs=[0.5, 1.0, 1.5],
                                directions=[0, 45, 90, 135]):
    best_energy = -1
    best_result = None
    results = []

    for fs in filter_sizes:
        for freq in freqs:
            for direction in directions:
                filt = create_cosine_filter(fs, freq, direction)
                conv = fftconvolve(image, filt, mode="same")
                energy = np.mean(conv**2)
                results.append({
                    "filter_size": fs, "freq": freq, "direction": direction,
                    "energy": energy, "conv": conv
                })
                if energy > best_energy:
                    best_energy = energy
                    best_result = conv

    return best_result, results


def visualize_filter_and_convolution(image, conv_result, filt, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Bolt Image")
    axes[0].axis("off")
    axes[1].imshow(filt, cmap="RdBu_r")
    axes[1].set_title(f"Cosine Filter{title_suffix}")
    axes[1].axis("off")
    axes[2].imshow(np.abs(conv_result), cmap="hot")
    axes[2].set_title("Convolution Result")
    axes[2].axis("off")
    plt.tight_layout()
    return fig


def get_image_files(data_dir, cat):
    cat_dir = os.path.join(data_dir, cat)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([f for f in os.listdir(cat_dir) if f.lower().endswith(exts)])


def visualize_sample_results(data_dir, output_dir, target_size=256):
    os.makedirs(output_dir, exist_ok=True)
    categories = ["threaded", "threadless"]

    for cat in categories:
        files = get_image_files(data_dir, cat)
        if not files:
            continue

        img_path = os.path.join(data_dir, cat, files[0])
        image = load_image(img_path, target_size=target_size)

        conv_result, results = detect_threads_convolution(image)

        best = max(results, key=lambda x: x["energy"])
        filt = create_cosine_filter(best["filter_size"], best["freq"], best["direction"])

        fig = visualize_filter_and_convolution(
            image, conv_result, filt,
            title_suffix=f"\n(freq={best['freq']}, dir={best['direction']})"
        )
        fig.suptitle(f"{cat.capitalize()} Bolt - Convolution Detection", fontsize=14)
        fig.savefig(os.path.join(output_dir, f"convolution_{cat}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved convolution_{cat}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, cat in enumerate(categories):
        files = get_image_files(data_dir, cat)
        if not files:
            continue
        img = load_image(os.path.join(data_dir, cat, files[0]), target_size=target_size)
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"{cat.capitalize()} Bolt")
        axes[idx].axis("off")
    plt.suptitle("Sample Bolt Images", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "sample_bolts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved sample_bolts.png")


def main():
    base_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, "dataset")
    output_dir = os.path.join(project_dir, "figures")

    print("=== Part 1, Step 1: 2D Cosine Convolution for Thread Detection ===\n")
    print(f"Using real dataset from: {data_dir}")
    print("Note: All images resized to a fixed 256x256 pixels. This normalises spatial")
    print("frequencies across varying input resolutions (512x512 to 3024x4032), ensuring")
    print("that cosine filter responses are comparable between images.\n")
    print("Generating visualizations...")
    visualize_sample_results(data_dir, output_dir, target_size=256)

    print("\nDone! Figures saved to figures/")


if __name__ == "__main__":
    main()
