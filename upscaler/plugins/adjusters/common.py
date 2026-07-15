"""Общие хелперы корректоров: нормализация dtype/альфы и светимость.

Существующие корректоры предполагают uint8; новые работают на float32 [0,1]
и обязаны возвращать dtype входа, сохраняя альфа-канал.
"""
import numpy as np

_EPS = 1e-6


def split_alpha(image: np.ndarray):
    """(rgb float32 [0,1], alpha (H,W,1) или None, was_uint8)."""
    was_uint8 = image.dtype == np.uint8
    arr = image.astype(np.float32)
    if was_uint8:
        arr = arr / 255.0
    alpha = None
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3:4]
        arr = arr[:, :, :3]
    return np.clip(arr, 0.0, 1.0), alpha, was_uint8


def merge_alpha(rgb: np.ndarray, alpha, was_uint8: bool) -> np.ndarray:
    out = np.clip(rgb, 0.0, 1.0)
    if alpha is not None:
        out = np.concatenate([out, alpha], axis=2)
    if was_uint8:
        return (out * 255.0 + 0.5).astype(np.uint8)
    return out.astype(np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2])


def scale_by_luminance(rgb: np.ndarray, l_old: np.ndarray,
                       l_new: np.ndarray) -> np.ndarray:
    """Перенести изменение светимости на RGB, сохраняя оттенок."""
    ratio = (l_new / np.maximum(l_old, _EPS))[..., None]
    return np.clip(rgb * ratio, 0.0, 1.0)
