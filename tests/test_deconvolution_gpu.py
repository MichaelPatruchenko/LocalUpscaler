import numpy as np
import pytest

torch = pytest.importorskip("torch")

import cv2
from upscaler.plugins.deblur.kernels import gaussian_kernel
from upscaler.engine.deconvolution import deconvolve
from upscaler.engine.deconvolution_gpu import deconvolve_torch


def _blurred():
    rng = np.random.default_rng(0)
    img = rng.random((96, 96)).astype(np.float64)
    img[24:72, 24:72] += 1.0
    img = np.clip(img, 0, 1)
    kernel = gaussian_kernel(2.0)
    blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return blurred, kernel


@pytest.mark.parametrize("method,atol", [
    ("wiener", 1e-6),
    ("tikhonov", 1e-6),
    ("rl", 1e-4),
    ("tv", 1e-4),
])
def test_torch_matches_numpy_on_cpu(method, atol):
    blurred, kernel = _blurred()
    cpu = deconvolve(blurred, kernel, method=method, smooth=30.0,
                     iterations=120, taper=True)
    gpu = deconvolve_torch(blurred, kernel, method=method, smooth=30.0,
                           iterations=120, taper=True, device="cpu")
    assert gpu.shape == cpu.shape
    assert np.allclose(gpu, cpu, atol=atol, rtol=1e-4)
