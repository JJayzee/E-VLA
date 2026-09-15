import numpy as np
from PIL import Image


def srgb_to_linear(image):
    # input: float32, (H,W,C), [0,1]
    # output: float32, (H,W,C), [0,1]
    a = 0.055
    mask = image <= 0.04045
    out = np.empty_like(image)
    out[mask] = image[mask] / 12.92
    out[~mask] = ((image[~mask] + a) / (1 + a)) ** 2.4
    return out

def linear_to_srgb(image):
    # input: float32, (H,W,C), [0,1]
    # output: float32, (H,W,C), [0,1]
    a = 0.055
    mask = image <= 0.0031308
    out = np.empty_like(image)
    out[mask] = image[mask] * 12.92
    out[~mask] = (1 + a) * (image[~mask] ** (1 / 2.4)) - a
    return out

def compute_white_balance_gains(image_uint8):
    img = image_uint8.astype(np.float32) / 255.0
    img = srgb_to_linear(img)
    ch_med = np.median(img.reshape(-1, 3), axis=0)
    gray = ch_med.mean()
    gains = gray / (ch_med + 1e-12)
    return gains

def apply_white_balance(image, gains):
    # input: uint8, (H,W,C)
    image = image.astype(np.float32) / 255.0
    image = srgb_to_linear(image)
    # lum estimation
    # lum = 0.2126 * image[..., 2] + 0.7152 * image[..., 1] + 0.0722 * image[..., 0]
    # green calibration, by experiment
    # lum_key = np.array([ 0.0, 0.4, 0.6,  0.7,  0.8,  0.84, 0.86, 0.9, 1.0])
    # adj_gain = np.array([1.0, 0.9, 0.72, 0.68, 0.72, 0.85, 1.0,  1.2, 1.3])
    # green_boost = np.interp(lum, lum_key, adj_gain)
    image *= gains.reshape(1, 1, 3)
    # image[..., 1] *= green_boost
    image = linear_to_srgb(image) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)

def save_image_and_events(image, events, frame_index, images_dir):
    image_path = images_dir / f"frame_{frame_index:04d}.png"
    img = Image.fromarray(image)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(image_path), quality=100)
    if events is not None:
        event_path = images_dir / f"events_{frame_index:04d}.npy"
        np.save(event_path, events)
