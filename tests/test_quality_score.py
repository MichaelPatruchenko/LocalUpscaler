"""Композитная оценка качества кадра (для blend-поиска)."""
import cv2
import numpy as np
from upscaler.engine.quality_score import quality_score


def _scene(size=256, seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (90, 110, 130)
    for _ in range(20):
        x, y = rng.integers(0, size - 40, 2)
        color = tuple(int(v) for v in rng.integers(30, 226, 3))
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 30, int(y) + 30),
                      color, -1)
    return img


def test_keys_and_ranges():
    out = quality_score(_scene())
    assert set(out) == {"score", "sharpness", "contrast", "colorfulness",
                        "exposure", "noise_penalty"}
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_blur_lowers_score_and_sharpness():
    img = _scene()
    blurred = cv2.GaussianBlur(img, (0, 0), 4.0)
    a, b = quality_score(img), quality_score(blurred)
    assert b["sharpness"] < a["sharpness"]
    assert b["score"] < a["score"]


def test_clipping_lowers_exposure():
    img = _scene()
    clipped = np.clip(img.astype(np.int32) + 150, 0, 255).astype(np.uint8)
    a, b = quality_score(img), quality_score(clipped)
    assert b["exposure"] < a["exposure"]


def test_grayscale_content_less_colorful():
    img = _scene()
    gray3 = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                         cv2.COLOR_GRAY2RGB)
    assert quality_score(gray3)["colorfulness"] < \
        quality_score(img)["colorfulness"]


def test_heavy_noise_lowers_noise_penalty():
    img = _scene()
    noisy = np.clip(img.astype(np.float64) +
                    np.random.default_rng(2).normal(0, 25, img.shape),
                    0, 255).astype(np.uint8)
    assert quality_score(noisy)["noise_penalty"] < \
        quality_score(img)["noise_penalty"]


def test_float_input_equivalent():
    img = _scene()
    a = quality_score(img)
    b = quality_score(img.astype(np.float32) / 255.0)
    assert abs(a["score"] - b["score"]) < 0.02
