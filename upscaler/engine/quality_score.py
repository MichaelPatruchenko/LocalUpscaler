"""Композитная алгоритмическая оценка качества кадра для blend-поиска.

Все компоненты нормированы в 0..1 (больше = лучше). Детерминирована.
"""
import cv2
import numpy as np

_WEIGHTS = {
    "sharpness": 0.30,
    "contrast": 0.20,
    "colorfulness": 0.15,
    "exposure": 0.20,
    "noise_penalty": 0.15,
}


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if a.size and a.max() <= 1.5:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def quality_score(img: np.ndarray) -> dict:
    """Оценка кадра: score (0..1) + разбивка по свойствам."""
    rgb = _to_uint8_rgb(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(min(lap.var() / 500.0, 1.0))

    contrast = float(min(gray.std() / 64.0, 1.0))

    # Colorfulness (Hasler–Süsstrunk): sigma_rgyb + 0.3*mu_rgyb, норма ~100.
    r, g, b = (rgb[..., i].astype(np.float64) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    cf = (np.sqrt(rg.std() ** 2 + yb.std() ** 2)
          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    colorfulness = float(min(cf / 100.0, 1.0))

    clip_frac = float(((gray <= 2) | (gray >= 253)).mean())
    exposure = float(max(0.0, 1.0 - 4.0 * clip_frac))

    sigma = float(np.sqrt(np.pi / 2) * np.mean(np.abs(lap)) / 6.0)
    noise_penalty = float(max(0.0, 1.0 - sigma / 20.0))

    parts = {
        "sharpness": sharpness, "contrast": contrast,
        "colorfulness": colorfulness, "exposure": exposure,
        "noise_penalty": noise_penalty,
    }
    score = sum(_WEIGHTS[k] * v for k, v in parts.items())
    return {"score": round(float(score), 4),
            **{k: round(v, 4) for k, v in parts.items()}}
