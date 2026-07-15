"""Shadows/Highlights: локальная коррекция через размытую маску светимости."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import (
    split_alpha, merge_alpha, luminance, scale_by_luminance,
)


class ShadowsHighlightsPlugin(BasePlugin):
    name = "Shadows/Highlights"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "shadows": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                    "default": 0.35, "ui": "slider"},
        "highlights": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "default": 0.2, "ui": "slider"},
        "radius": {"type": "integer", "minimum": 3, "maximum": 100,
                   "default": 30, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        shadows = float(params.get("shadows", 0.35))
        highlights = float(params.get("highlights", 0.2))
        radius = int(params.get("radius", 30))
        if shadows <= 0.0 and highlights <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        lum = luminance(rgb)
        blurred = cv2.GaussianBlur(lum, (0, 0), max(radius / 3.0, 1.0))
        sh_w = np.clip(1.0 - blurred / 0.5, 0.0, 1.0) ** 1.5
        hi_w = np.clip((blurred - 0.5) / 0.5, 0.0, 1.0) ** 1.5
        l_new = (lum
                 + shadows * sh_w * (np.power(lum, 0.55) - lum)
                 + highlights * hi_w * (np.power(lum, 1.8) - lum))
        out = scale_by_luminance(rgb, lum, l_new)
        return merge_alpha(out, alpha, was_u8)
