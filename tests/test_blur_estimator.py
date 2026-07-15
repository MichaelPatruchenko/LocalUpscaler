import cv2
import numpy as np
import pytest
from upscaler.plugins.deblur.kernels import (
    motion_kernel, gaussian_kernel, focus_kernel,
)
from upscaler.engine.blur_estimator import BlurEstimator


def _scene(size=256):
    rng = np.random.default_rng(1)
    img = (rng.random((size, size)) * 80 + 60)
    img[40:200, 40:200] += 100
    cv2.rectangle(img, (80, 80), (160, 160), 220, -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_estimate_returns_expected_keys():
    est = BlurEstimator()
    out = est.estimate(_scene())
    assert set(["blur_type", "radius", "angle", "smooth"]).issubset(out)
    assert out["blur_type"] in ("focus", "motion", "gaussian")
    assert out["radius"] > 0


def test_detects_horizontal_motion_angle():
    est = BlurEstimator()
    scene = _scene().astype(np.float64) / 255.0
    blurred = cv2.filter2D(scene, -1, motion_kernel(8.0, angle=0.0))
    blurred_u8 = np.clip(blurred * 255, 0, 255).astype(np.uint8)
    out = est.estimate(blurred_u8)
    assert out["blur_type"] == "motion"
    ang = out["angle"] % 180
    assert min(ang, 180 - ang) < 20


def test_defocus_is_not_classified_as_motion():
    est = BlurEstimator()
    scene = _scene().astype(np.float64) / 255.0
    blurred = cv2.filter2D(scene, -1, gaussian_kernel(4.0))
    blurred_u8 = np.clip(blurred * 255, 0, 255).astype(np.uint8)
    out = est.estimate(blurred_u8)
    # smooth gaussian falloff -> classified as gaussian, never motion
    assert out["blur_type"] == "gaussian"


def test_disk_blur_classified_as_focus():
    est = BlurEstimator()
    scene = _scene().astype(np.float64) / 255.0
    blurred = cv2.filter2D(scene, -1, focus_kernel(7.0))
    blurred_u8 = np.clip(blurred * 255, 0, 255).astype(np.uint8)
    out = est.estimate(blurred_u8)
    assert out["blur_type"] == "focus"
    assert out["radius"] > 0


def test_accepts_float_normalized_input():
    # float [0,1] input must not break Canny-based radius estimation
    est = BlurEstimator()
    scene = _scene().astype(np.float64) / 255.0
    blurred = cv2.filter2D(scene, -1, gaussian_kernel(3.0))
    out = est.estimate(blurred.astype(np.float32))
    assert out["blur_type"] in ("focus", "gaussian")
    assert out["radius"] >= 1.0  # never below the radius floor


def test_noise_increases_smooth():
    est = BlurEstimator()
    clean = _scene()
    noisy = np.clip(clean.astype(np.float64) +
                    np.random.default_rng(2).normal(0, 25, clean.shape),
                    0, 255).astype(np.uint8)
    s_clean = est.estimate(clean)["smooth"]
    s_noisy = est.estimate(noisy)["smooth"]
    assert s_noisy >= s_clean


def _sharp_detail(size=256):
    # High-frequency content: random noise has energy across all frequencies.
    rng = np.random.default_rng(7)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def test_assess_sharp_image_does_not_need_deblur():
    est = BlurEstimator()
    a = est.assess(_sharp_detail())
    assert a["needs_deblur"] is False
    assert a["sharpness"] > 0.5
    # assess() must preserve the estimate() fields
    assert set(["blur_type", "radius", "angle", "smooth"]).issubset(a)


def test_assess_defocus_blur_needs_deblur():
    est = BlurEstimator()
    sharp = _sharp_detail()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4.0)
    a = est.assess(blurred)
    assert a["needs_deblur"] is True
    assert a["sharpness"] < est.assess(sharp)["sharpness"]


def test_assess_method_tv_for_strong_blur():
    est = BlurEstimator()
    sharp = _sharp_detail()
    strong = cv2.GaussianBlur(sharp, (0, 0), 9.0)
    a = est.assess(strong)
    # large estimated radius -> Total Variation
    assert a["method"] in ("tv", "tikhonov", "wiener")
    if a["radius"] >= 8.0:
        assert a["method"] == "tv"


def test_assess_method_tikhonov_when_noisy():
    est = BlurEstimator()
    rng = np.random.default_rng(3)
    base = cv2.GaussianBlur(_sharp_detail(), (0, 0), 3.0).astype(np.float64)
    noisy = np.clip(base + rng.normal(0, 40, base.shape), 0, 255).astype(np.uint8)
    a = est.assess(noisy)
    assert a["method"] in ("tikhonov", "tv", "wiener")


# --- Этап 1: полный набор параметров деконволюции ---------------------------

def test_extra_params_tv_iterations_only_for_tv_rl():
    tv = BlurEstimator._extra_params("gaussian", 9.0, "tv", 30.0, snr=10.0, dip=0.0)
    rl = BlurEstimator._extra_params("gaussian", 9.0, "rl", 30.0, snr=10.0, dip=0.0)
    wiener = BlurEstimator._extra_params("gaussian", 3.0, "wiener", 30.0, 10.0, 0.0)
    assert 60 <= tv["tv_iterations"] <= 400
    assert 60 <= rl["tv_iterations"] <= 400
    assert "tv_iterations" not in wiener


def test_extra_params_iterations_grow_with_radius():
    small = BlurEstimator._extra_params("gaussian", 8.0, "tv", 30.0, 10.0, 0.0)
    big = BlurEstimator._extra_params("gaussian", 15.0, "tv", 30.0, 10.0, 0.0)
    assert big["tv_iterations"] >= small["tv_iterations"]


def test_extra_params_focus_gets_feather_and_correction():
    focus = BlurEstimator._extra_params("focus", 5.0, "wiener", 30.0, 10.0, dip=0.8)
    gauss = BlurEstimator._extra_params("gaussian", 5.0, "wiener", 30.0, 10.0, 0.8)
    assert 5.0 <= focus["edge_feather"] <= 25.0
    assert 0.0 <= focus["correction_strength"] <= 50.0
    assert "edge_feather" not in gauss and "correction_strength" not in gauss


def test_extra_params_deep_dip_means_less_feather():
    weak = BlurEstimator._extra_params("focus", 5.0, "wiener", 30.0, 10.0, dip=0.1)
    deep = BlurEstimator._extra_params("focus", 5.0, "wiener", 30.0, 10.0, dip=1.0)
    assert deep["edge_feather"] < weak["edge_feather"]
    assert deep["correction_strength"] > weak["correction_strength"]


def test_extra_params_low_snr_raises_smooth():
    clean = BlurEstimator._extra_params("gaussian", 3.0, "wiener", 30.0, snr=25.0, dip=0.0)
    noisy = BlurEstimator._extra_params("gaussian", 3.0, "wiener", 30.0, snr=2.0, dip=0.0)
    assert noisy["smooth"] > clean["smooth"]
    assert 20.0 <= noisy["smooth"] <= 60.0
    assert clean["edge_taper"] is True


def test_assess_includes_extra_params():
    est = BlurEstimator()
    scene = _scene().astype(np.float64) / 255.0
    blurred = cv2.filter2D(scene, -1, focus_kernel(7.0))
    blurred_u8 = np.clip(blurred * 255, 0, 255).astype(np.uint8)
    out = est.assess(blurred_u8)
    assert out["edge_taper"] is True
    assert "smooth" in out
    if out["blur_type"] == "focus":
        assert "edge_feather" in out and "correction_strength" in out
    if out["method"] in ("tv", "rl"):
        assert "tv_iterations" in out
