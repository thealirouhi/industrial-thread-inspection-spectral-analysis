import numpy as np
from PIL import Image
import os


def generate_bolt_image(size=256, threaded=True, rotation=0, scale=1.0):
    img = np.zeros((size, size), dtype=np.float64)
    cx, cy = size // 2, size // 2
    Y, X = np.mgrid[:size, :size]
    dx = X - cx
    dy = Y - cy
    theta = np.radians(rotation)
    rx = dx * np.cos(theta) + dy * np.sin(theta)
    ry = -dx * np.sin(theta) + dy * np.cos(theta)
    bolt_radius = size * 0.35 * scale
    head_radius = size * 0.42 * scale
    bolt_mask = (rx**2 + (ry / 4.0)**2) < bolt_radius**2
    head_mask = ((rx**2 + ry**2) < head_radius**2) & (np.abs(ry) > bolt_radius * 0.8)
    img[bolt_mask] = 0.7
    img[head_mask] = 0.85
    if threaded:
        thread_freq = 12.0 / scale
        thread_depth = 0.15
        thread_mask = bolt_mask & (~head_mask)
        thread_pattern = thread_depth * np.sin(thread_freq * ry / bolt_radius)
        img[thread_mask] += thread_pattern[thread_mask]
        ridge_mask = thread_mask & (np.abs(np.sin(thread_freq * ry / bolt_radius)) > 0.7)
        img[ridge_mask] += 0.08
    img = np.clip(img, 0, 1)
    noise = np.random.normal(0, 0.02, img.shape)
    img = np.clip(img + noise, 0, 1)
    return img


def generate_dataset(output_dir, num_images=10, size=256):
    threaded_dir = os.path.join(output_dir, "threaded")
    unthreaded_dir = os.path.join(output_dir, "unthreaded")
    os.makedirs(threaded_dir, exist_ok=True)
    os.makedirs(unthreaded_dir, exist_ok=True)

    for i in range(num_images):
        rotation = np.random.uniform(-30, 30)
        scale = np.random.uniform(0.7, 1.0)
        offset_x = np.random.uniform(-10, 10)
        offset_y = np.random.uniform(-10, 10)

        img = generate_bolt_image(size, threaded=True, rotation=rotation, scale=scale)
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(threaded_dir, f"bolt_{i:03d}.png"))

        img = generate_bolt_image(size, threaded=False, rotation=rotation, scale=scale)
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(os.path.join(unthreaded_dir, f"bolt_{i:03d}.png"))

    print(f"Generated {num_images} threaded and {num_images} unthreaded bolt images in {output_dir}")


if __name__ == "__main__":
    np.random.seed(42)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    generate_dataset(data_dir, num_images=10, size=256)
