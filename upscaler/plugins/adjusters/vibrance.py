"""Vibrance: умная насыщенность — тусклым больше, коже меньше."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import split_alpha, merge_alpha


class VibrancePlugin(BasePlugin):
    name = "Vibrance"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": -1.0, "maximum": 1.0,
                     "default": 0.4, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.4))
        if abs(strength) < 1e-3:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        hsv = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h = hsv[..., 0].astype(np.float32)          # 0..179
        s = hsv[..., 1].astype(np.float32) / 255.0
        v = hsv[..., 2].astype(np.float32) / 255.0
        # Кожные тона (оранжевые hue) защищаются.
        skin = ((h >= 5) & (h <= 25) & (s > 0.1) & (v > 0.2))
        weight = np.where(skin, 0.3, 1.0)
        if strength >= 0:
            s_new = s * (1.0 + strength * (1.0 - s) ** 2 * weight)
        else:
            s_new = s * (1.0 + strength * weight * 0.8)
        hsv[..., 1] = np.clip(s_new * 255.0, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return merge_alpha(out, alpha, was_u8)
