"""Optics: коррекция виньетки (радиальная) и хроматических аберраций."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import split_alpha, merge_alpha


class OpticsPlugin(BasePlugin):
    name = "Optics"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "vignette": {"type": "number", "minimum": -1.0, "maximum": 1.0,
                     "default": 0.3, "ui": "slider"},
        "ca": {"type": "number", "minimum": 0.0, "maximum": 1.0,
               "default": 0.5, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    @staticmethod
    def _scale_channel(ch: np.ndarray, scale: float) -> np.ndarray:
        h, w = ch.shape
        cx, cy = w / 2.0, h / 2.0
        m = np.float32([[scale, 0, cx * (1 - scale)],
                        [0, scale, cy * (1 - scale)]])
        return cv2.warpAffine(ch, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        vignette = float(params.get("vignette", 0.3))
        ca = float(params.get("ca", 0.5))
        if abs(vignette) < 1e-3 and ca < 1e-3:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        h, w = rgb.shape[:2]
        if abs(vignette) >= 1e-3:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            r2 = (((xx - w / 2) / (w / 2)) ** 2
                  + ((yy - h / 2) / (h / 2)) ** 2) / 2.0
            gain = 1.0 + 0.5 * vignette * r2[..., None]
            rgb = np.clip(rgb * gain, 0.0, 1.0)
        if ca >= 1e-3:
            shift = 1.0 + 0.002 * ca
            rgb = np.stack([
                self._scale_channel(rgb[..., 0], shift),
                rgb[..., 1],
                self._scale_channel(rgb[..., 2], 1.0 / shift),
            ], axis=2)
        return merge_alpha(rgb, alpha, was_u8)
