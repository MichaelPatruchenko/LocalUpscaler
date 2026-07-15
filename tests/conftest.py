"""Shared test fixtures."""

import numpy as np
import pytest
from pathlib import Path


def pytest_configure(config):
    """Ensure the configured --basetemp parent dir exists.

    pytest.ini pins --basetemp to an ASCII path (to avoid OpenCV failures on
    machines with non-ASCII usernames), but pytest's mkdir is non-recursive,
    so the parent must exist first. Creating it here keeps tests runnable
    out-of-the-box on a fresh checkout.
    """
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp:
        Path(basetemp).parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_rgb_uint8():
    """Small 64x64 RGB uint8 test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_rgb_float32():
    """Small 64x64 RGB float32 test image in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((64, 64, 3), dtype=np.float32)


@pytest.fixture
def sample_gray_uint8():
    """Small 64x64 grayscale uint8 test image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def tmp_image_path(tmp_path, sample_rgb_uint8):
    """Write a test PNG and return its path."""
    import cv2
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), cv2.cvtColor(sample_rgb_uint8, cv2.COLOR_RGB2BGR))
    return p
