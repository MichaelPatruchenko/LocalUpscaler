"""SmartDeblur plugin: blind/manual FFT deconvolution for focus and motion blur."""
import logging

import cv2
import numpy as np

from upscaler.plugins.base import BasePlugin
from upscaler.plugins.deblur.kernels import (
    focus_kernel, motion_kernel, gaussian_kernel,
)
from upscaler.engine.deconvolution import deconvolve
from upscaler.engine.blur_estimator import BlurEstimator

log = logging.getLogger(__name__)

# Quality safeguard thresholds. Deblur is kept only if it makes the image
# genuinely sharper without heavy ringing; otherwise the original is returned.
_MIN_SHARPEN_GAIN = 1.05     # Laplacian-variance(after)/before must exceed this
_MAX_OVERSHOOT_FRAC = 0.02   # max fraction of pre-clamp pixels outside [0,1]
_OVERSHOOT_EPS = 0.02


def _to_gray_f(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr[:, :, :3].mean(axis=2)
    return arr


def _deblur_quality_ok(before: np.ndarray, after: np.ndarray,
                       pre_clamp: np.ndarray) -> bool:
    """True if `after` is sharper than `before` without excessive ringing.

    before/after are float [0,1]; pre_clamp is the result before clipping to
    [0,1] (used to measure overshoot/ringing)."""
    gb = _to_gray_f(before)
    ga = _to_gray_f(after)
    vb = cv2.Laplacian(gb, cv2.CV_64F).var()
    va = cv2.Laplacian(ga, cv2.CV_64F).var()
    gain = va / (vb + 1e-9)
    overshoot = float(np.mean((pre_clamp < -_OVERSHOOT_EPS)
                              | (pre_clamp > 1.0 + _OVERSHOOT_EPS)))
    return bool(gain >= _MIN_SHARPEN_GAIN and overshoot <= _MAX_OVERSHOOT_FRAC)


class SmartDeblurPlugin(BasePlugin):
    name = "SmartDeblur"
    category = "deblur"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "auto": {"type": "boolean", "default": True},
        "blur_type": {
            "type": "string", "ui": "combo", "default": "gaussian",
            "options": ["focus", "motion", "gaussian"],
            "labels": {"focus": "Расфокус", "motion": "Смаз",
                       "gaussian": "Гаусс"},
        },
        "radius": {"type": "number", "minimum": 0.1, "maximum": 50.0,
                   "default": 3.0, "ui": "slider"},
        "angle": {"type": "number", "minimum": 0.0, "maximum": 180.0,
                  "default": 0.0, "ui": "slider"},
        "smooth": {"type": "number", "minimum": 1.0, "maximum": 100.0,
                   "default": 30.0, "ui": "slider"},
        "edge_feather": {"type": "number", "minimum": 0.0, "maximum": 100.0,
                         "default": 10.0, "ui": "slider"},
        "correction_strength": {"type": "number", "minimum": 0.0,
                                "maximum": 100.0, "default": 0.0, "ui": "slider"},
        "method": {
            "type": "string", "ui": "combo", "default": "wiener",
            "options": ["wiener", "tikhonov", "tv", "rl"],
            "labels": {"wiener": "Wiener", "tikhonov": "Tikhonov",
                       "tv": "Total Variation", "rl": "Richardson-Lucy"},
        },
        "tv_iterations": {"type": "integer", "minimum": 10, "maximum": 1000,
                          "default": 300, "ui": "slider"},
        "edge_taper": {"type": "boolean", "default": True},
    }

    def __init__(self):
        self._estimator = BlurEstimator()
        self.device = "cpu"

    def initialize(self, device: str) -> None:
        self.device = device

    def _build_kernel(self, params: dict) -> np.ndarray:
        blur_type = params.get("blur_type", "gaussian")
        radius = float(params.get("radius", 3.0))
        if blur_type == "motion":
            return motion_kernel(radius, float(params.get("angle", 0.0)))
        if blur_type == "focus":
            return focus_kernel(
                radius,
                float(params.get("edge_feather", 10.0)),
                float(params.get("correction_strength", 0.0)),
            )
        return gaussian_kernel(radius)

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        is_uint8 = image.dtype == np.uint8
        # Work in float64 for FFT/iterative numerical stability (TV in
        # particular). uint8 is scaled to [0,1]; float input is assumed to be
        # in [0,1] per the BasePlugin contract and clamped defensively so a
        # stray out-of-range image cannot corrupt blur estimation.
        if is_uint8:
            work = image.astype(np.float64) / 255.0
        else:
            work = image.astype(np.float64)
            if work.size and (work.max() > 1.0 + 1e-3 or work.min() < -1e-3):
                log.warning("SmartDeblur: float input outside [0,1], clamping")
                work = np.clip(work, 0.0, 1.0)

        params = dict(params or {})
        if params.get("auto", True):
            gray = (np.clip(work, 0, 1) * 255.0).astype(np.uint8)
            estimated = self._estimator.estimate(gray)
            for key, val in estimated.items():
                params.setdefault(key, val)
            log.info("SmartDeblur auto params: %s", estimated)

        kernel = self._build_kernel(params)
        method = params.get("method", "wiener")
        smooth = float(params.get("smooth", 30.0))
        iterations = int(params.get("tv_iterations", 300))
        taper = bool(params.get("edge_taper", True))
        progress_cb = params.get("_progress_cb")
        cancel_cb = params.get("_cancel_cb")

        if work.ndim == 2:
            channels = [work]
        else:
            channels = [work[:, :, c] for c in range(work.shape[2])]

        out_channels = []
        for ch in channels[:3]:
            restored = deconvolve(ch, kernel, method=method, smooth=smooth,
                                  iterations=iterations, taper=taper,
                                  progress_cb=progress_cb, cancel_cb=cancel_cb,
                                  device=self.device)
            out_channels.append(restored)

        if work.ndim == 3 and work.shape[2] > 3:
            out_channels.extend(work[:, :, c] for c in range(3, work.shape[2]))

        pre_clamp = (out_channels[0] if work.ndim == 2
                     else np.stack(out_channels, axis=2))
        result = np.clip(pre_clamp, 0.0, 1.0)

        # Quality safeguard: keep the deblur only if it genuinely improves the
        # image; otherwise return the original unchanged (deblur skipped).
        if not _deblur_quality_ok(work, result, pre_clamp):
            log.info("SmartDeblur: result not an improvement; keeping original")
            return image

        if is_uint8:
            return (result * 255.0).round().astype(np.uint8)
        return result.astype(np.float32)
