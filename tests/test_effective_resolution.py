"""Оценка эффективного разрешения (спектральная, по кропам оригинала)."""
import cv2
import numpy as np
from upscaler.engine.effective_resolution import (
    FACTORS, effective_downscale_factor,
)


def _scene(size=512, seed=1, noise=3.0):
    """Реалистичная сцена: градиенты + резкие фигуры + умеренный шум."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = 90 + 50 * np.sin(xx / 97.0) + 40 * np.cos(yy / 61.0)
    for _ in range(int(25 * (size / 512) ** 2)):
        x, y = rng.integers(0, size - 60, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 40, int(y) + 40),
                      int(rng.integers(40, 220)), -1)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _upsampled(img, factor):
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * factor), int(h * factor)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def test_sharp_image_keeps_full_resolution():
    assert effective_downscale_factor(_scene()) == 1.0


def test_noisy_sharp_image_protected():
    assert effective_downscale_factor(_scene(seed=4, noise=8.0)) == 1.0


def test_up2x_detected_as_half():
    assert effective_downscale_factor(_upsampled(_scene(), 0.5)) == 0.5


def test_up3x_detected_at_most_half():
    f = effective_downscale_factor(_upsampled(_scene(seed=2), 1 / 3))
    assert f <= 0.5


def test_large_image_multi_crop():
    big_sharp = _scene(1400, seed=9)
    big_soft = _upsampled(_scene(1400, seed=10), 0.5)
    assert effective_downscale_factor(big_sharp) == 1.0
    assert effective_downscale_factor(big_soft) == 0.5


def test_tiny_image_returns_one():
    assert effective_downscale_factor(np.zeros((24, 24), np.uint8)) == 1.0


def test_float_and_color_inputs():
    img = _upsampled(_scene(), 0.5)
    color = np.stack([img] * 3, axis=2)
    as_float = img.astype(np.float32) / 255.0
    assert effective_downscale_factor(color) == 0.5
    assert effective_downscale_factor(as_float) == 0.5


def test_factors_sorted_descending():
    assert list(FACTORS) == sorted(FACTORS, reverse=True)
