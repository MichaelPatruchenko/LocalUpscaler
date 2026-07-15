"""Color space conversions and ICC profile handling."""

import numpy as np


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert sRGB uint8/uint16 image to linear float32 [0, 1]."""
    f = normalize_to_float(image)
    # sRGB gamma decoding
    mask = f <= 0.04045
    result = np.where(mask, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)
    return result.astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert linear float32 [0, 1] to sRGB uint8."""
    img = np.clip(image, 0.0, 1.0)
    mask = img <= 0.0031308
    result = np.where(mask, img * 12.92, 1.055 * np.power(img, 1.0 / 2.4) - 0.055)
    return (np.clip(result, 0.0, 1.0) * 255).astype(np.uint8)


def normalize_to_float(image: np.ndarray) -> np.ndarray:
    """Normalize any image to float32 [0, 1]."""
    if image.dtype == np.float32 or image.dtype == np.float64:
        return image.astype(np.float32)
    elif image.dtype == np.uint8:
        return (image / 255.0).astype(np.float32)
    elif image.dtype == np.uint16:
        return (image / 65535.0).astype(np.float32)
    raise ValueError(f"Unsupported dtype: {image.dtype}")


def denormalize_to_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert float32 [0, 1] to target dtype."""
    img = np.clip(image, 0.0, 1.0)
    if dtype == np.uint8:
        return (img * 255).astype(np.uint8)
    elif dtype == np.uint16:
        return (img * 65535).astype(np.uint16)
    elif dtype == np.float32:
        return img.astype(np.float32)
    raise ValueError(f"Unsupported target dtype: {dtype}")


def embed_icc_profile(path, icc_bytes: bytes):
    """Embed ICC profile into an image file via Pillow."""
    if icc_bytes is None:
        return
    from PIL import Image
    img = Image.open(path)
    img.save(path, icc_profile=icc_bytes)
