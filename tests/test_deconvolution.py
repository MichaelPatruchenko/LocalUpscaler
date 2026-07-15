import cv2
import numpy as np
import pytest
from upscaler.plugins.deblur.kernels import gaussian_kernel, motion_kernel
from upscaler.engine.deconvolution import (
    wiener_deconvolve, tikhonov_deconvolve, tv_deconvolve, deconvolve,
    edge_taper, richardson_lucy,
)


def _sharp_image(size=128):
    rng = np.random.default_rng(0)
    img = rng.random((size, size)).astype(np.float64)
    img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] += 1.0
    return np.clip(img, 0, 1)


def _blur_with(img, kernel):
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def _sharpness(img):
    return cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var()


@pytest.mark.parametrize("method", ["wiener", "tikhonov"])
def test_deconvolution_increases_sharpness(method):
    sharp = _sharp_image()
    kernel = gaussian_kernel(2.5)
    blurred = _blur_with(sharp, kernel)
    restored = deconvolve(blurred, kernel, method=method, smooth=30)
    assert restored.shape == blurred.shape
    assert np.isfinite(restored).all()
    assert _sharpness(restored) > _sharpness(blurred)


def test_tv_runs_and_increases_sharpness():
    sharp = _sharp_image(96)
    kernel = gaussian_kernel(2.0)
    blurred = _blur_with(sharp, kernel)
    restored = tv_deconvolve(blurred, kernel, smooth=30, iterations=40)
    assert np.isfinite(restored).all()
    assert _sharpness(restored) > _sharpness(blurred)


def test_dispatcher_methods_and_fallback():
    sharp = _sharp_image(64)
    kernel = gaussian_kernel(2.0)
    blurred = _blur_with(sharp, kernel)
    tv = deconvolve(blurred, kernel, method="tv", smooth=30, iterations=20)
    assert tv.shape == blurred.shape and np.isfinite(tv).all()
    # unknown method falls back to wiener (compare with taper disabled so the
    # bare wiener_deconvolve matches the dispatcher's routing exactly)
    fallback = deconvolve(blurred, kernel, method="nope", smooth=30, taper=False)
    wiener = wiener_deconvolve(blurred, kernel, smooth=30)
    assert np.allclose(fallback, wiener)


def test_edge_taper_preserves_shape_and_interior():
    sharp = _sharp_image()
    kernel = gaussian_kernel(3.0)
    tapered = edge_taper(sharp, kernel)
    assert tapered.shape == sharp.shape
    assert np.isfinite(tapered).all()
    # interior is essentially untouched (alpha ~ 1 there)
    cy, cx = sharp.shape[0] // 2, sharp.shape[1] // 2
    assert abs(tapered[cy, cx] - sharp[cy, cx]) < 1e-3


def test_edge_taper_reduces_border_ringing():
    sharp = _sharp_image()
    kernel = gaussian_kernel(3.0)
    blurred = _blur_with(sharp, kernel)
    plain = deconvolve(blurred, kernel, method="wiener", smooth=20, taper=False)
    tapered = deconvolve(blurred, kernel, method="wiener", smooth=20, taper=True)
    # measure extreme overshoot in a thin border band
    def border_extreme(a):
        b = np.concatenate([a[:4, :].ravel(), a[-4:, :].ravel(),
                            a[:, :4].ravel(), a[:, -4:].ravel()])
        return b.std()
    assert border_extreme(tapered) <= border_extreme(plain)


def test_richardson_lucy_increases_sharpness():
    sharp = _sharp_image(96)
    kernel = gaussian_kernel(2.0)
    blurred = _blur_with(sharp, kernel)
    restored = richardson_lucy(blurred, kernel, iterations=20)
    assert np.isfinite(restored).all()
    assert _sharpness(restored) > _sharpness(blurred)


def test_dispatcher_rl_method():
    sharp = _sharp_image(64)
    kernel = gaussian_kernel(2.0)
    blurred = _blur_with(sharp, kernel)
    out = deconvolve(blurred, kernel, method="rl", smooth=30, iterations=200)
    assert out.shape == blurred.shape and np.isfinite(out).all()


def test_tv_cancel_stops_early():
    sharp = _sharp_image(64)
    kernel = gaussian_kernel(2.0)
    blurred = _blur_with(sharp, kernel)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    restored = tv_deconvolve(blurred, kernel, smooth=30, iterations=500,
                             cancel_cb=cancel)
    assert np.isfinite(restored).all()


def test_deconvolve_falls_back_to_numpy_when_gpu_raises(monkeypatch):
    pytest.importorskip("torch")  # test pokes dc.torch.cuda; skip on torch-less installs
    sharp = _sharp_image(64)
    kernel = gaussian_kernel(2.0)
    blurred = _blur_with(sharp, kernel)

    # Force the GPU branch to be attempted, then fail.
    import upscaler.engine.deconvolution as dc

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(dc.torch.cuda, "is_available", _FakeCuda.is_available, raising=False)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated GPU failure")

    import upscaler.engine.deconvolution_gpu as dcg
    monkeypatch.setattr(dcg, "deconvolve_torch", _boom)

    cpu = dc.deconvolve(blurred, kernel, method="wiener", smooth=30.0)
    fell_back = dc.deconvolve(blurred, kernel, method="wiener", smooth=30.0,
                              device="cuda")
    assert np.allclose(fell_back, cpu)
