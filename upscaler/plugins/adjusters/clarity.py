"""Clarity: локальный контраст (unsharp большого радиуса) с защитой краёв
диапазона."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import (
    split_alpha, merge_alpha, luminance, scale_by_luminance,
)


class ClarityPlugin(BasePlugin):
    name = "Clarity"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.3, "ui": "slider"},
        "radius": {"type": "integer", "minimum": 10, "maximum": 200,
                   "default": 60, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.3))
        radius = int(params.get("radius", 60))
        if strength <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        lum = luminance(rgb)
        low = cv2.GaussianBlur(lum, (0, 0), max(radius / 3.0, 1.0))
        high = lum - low
        protect = 4.0 * lum * (1.0 - lum)      # пик в midtones, 0 на краях
        l_new = np.clip(lum + strength * 1.2 * high * protect, 0.0, 1.0)
        out = scale_by_luminance(rgb, lum, l_new)
        return merge_alpha(out, alpha, was_u8)
