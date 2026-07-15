"""Blur kernel (PSF) builders ported from SmartDeblur's ImageUtils.cpp.

All builders return a normalized float32 2D array summing to 1.0.
"""
import cv2
import numpy as np

_SUPERSAMPLE = 4  # anti-aliasing factor for disk/line kernels


def _even_size(value: float, pad: int = 6) -> int:
    # pad >= 6 guarantees an even size >= 6, which keeps the supersample
    # reshape (size, ss, size, ss) valid for all callers.
    size = int(value + pad)
    size += size % 2
    return size


def _normalize(kernel: np.ndarray) -> np.ndarray:
    total = float(np.abs(kernel).sum())
    if total == 0:
        total = 1.0
    return (kernel / total).astype(np.float32)


def gaussian_kernel(radius: float) -> np.ndarray:
    radius = max(0.3, float(radius))
    size = _even_size(3.5 * radius)
    c = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    g = np.exp(-(((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * radius ** 2)))
    return _normalize(g)


def focus_kernel(radius: float, edge_feather: float = 10.0,
                 correction_strength: float = 0.0) -> np.ndarray:
    radius = max(0.5, float(radius))
    correction_strength = float(np.clip(correction_strength, -100.0, 100.0))
    edge_feather = float(max(edge_feather, 0.0))
    size = _even_size(2.0 * radius)
    ss = _SUPERSAMPLE
    big = size * ss
    yy, xx = np.ogrid[:big, :big]
    center_big = big / 2.0
    dist_big = np.sqrt((xx - center_big) ** 2 + (yy - center_big) ** 2) / ss
    disk = (dist_big <= radius).astype(np.float64)
    kernel = disk.reshape(size, ss, size, ss).mean(axis=(1, 3))

    if correction_strength != 0:
        center = size / 2.0
        yk, xk = np.ogrid[:size, :size]
        d = np.sqrt((xk - center) ** 2 + (yk - center) ** 2)
        mu = radius
        sigma = max(radius * edge_feather / 100.0, 1e-6)
        gauss = np.exp(-(((d - mu) / sigma) ** 2) / 2.0) * (correction_strength / 100.0)
        inside = d <= radius
        if correction_strength >= 0:
            kernel[inside] *= (100.0 - correction_strength) / 100.0
        kernel[inside] += gauss[inside]
        kernel = np.clip(kernel, 0.0, 1.0)

    return _normalize(kernel)


def motion_kernel(radius: float, angle: float) -> np.ndarray:
    length = max(1.0, 2.0 * float(radius))
    size = _even_size(length)
    ss = _SUPERSAMPLE
    big = size * ss
    canvas = np.zeros((big, big), dtype=np.uint8)
    angle_rad = np.deg2rad(float(angle))
    center = big / 2.0
    dx = np.cos(angle_rad) * length * ss / 2.0
    dy = np.sin(angle_rad) * length * ss / 2.0
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(canvas, p1, p2, color=255, thickness=ss, lineType=cv2.LINE_AA)
    kernel = canvas.astype(np.float64).reshape(size, ss, size, ss).mean(axis=(1, 3))
    return _normalize(kernel)
