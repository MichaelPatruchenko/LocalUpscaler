"""Auto Levels: процентильная растяжка + S-кривая по узости гистограммы."""
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import (
    split_alpha, merge_alpha, luminance,
)


class AutoLevelsPlugin(BasePlugin):
    name = "Auto Levels"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.5, "ui": "slider"},
        "clip_percent": {"type": "number", "minimum": 0.0, "maximum": 5.0,
                         "default": 0.5, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.5))
        clip = float(params.get("clip_percent", 0.5))
        if strength <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        lum = luminance(rgb)
        lo = float(np.percentile(lum, clip))
        hi = float(np.percentile(lum, 100.0 - clip))
        if hi - lo < 1e-3:
            return image
        stretched = np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)
        # S-кривая: сила от узости исходной гистограммы.
        narrowness = float(np.clip((1.0 - (hi - lo)) * 1.5, 0.0, 1.0))
        a = 0.35 * strength * narrowness
        curved = stretched - a * np.sin(2.0 * np.pi * stretched) / (2.0 * np.pi)
        out = rgb * (1.0 - strength) + curved * strength
        return merge_alpha(out, alpha, was_u8)
