"""Dodge & Burn: мягкие кривые через luminosity-маски теней и светов."""
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import (
    split_alpha, merge_alpha, luminance, scale_by_luminance,
)


class DodgeBurnPlugin(BasePlugin):
    name = "Dodge & Burn"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.3, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.3))
        if strength <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        lum = luminance(rgb)
        w_sh = np.clip(1.0 - lum / 0.4, 0.0, 1.0)
        w_hi = np.clip((lum - 0.6) / 0.4, 0.0, 1.0)
        w_mid = np.clip(1.0 - w_sh - w_hi, 0.0, 1.0)
        l_new = (lum
                 + strength * 0.15 * w_sh * (1.0 - lum)     # dodge теней
                 - strength * 0.12 * w_hi * lum             # burn светов
                 + strength * 0.10 * w_mid * (lum - 0.5))   # midtone contrast
        out = scale_by_luminance(rgb, lum, np.clip(l_new, 0.0, 1.0))
        return merge_alpha(out, alpha, was_u8)
