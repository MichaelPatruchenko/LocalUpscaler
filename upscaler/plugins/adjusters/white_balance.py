"""White Balance: gray-point по нейтральным областям, фолбэк gray-world."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import split_alpha, merge_alpha


class WhiteBalancePlugin(BasePlugin):
    name = "White Balance"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.7, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.7))
        if strength <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        hsv = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV)
        s = hsv[..., 1].astype(np.float32) / 255.0
        v = hsv[..., 2].astype(np.float32) / 255.0
        neutral = (s < 0.15) & (v > 0.2) & (v < 0.95)
        if neutral.mean() > 0.005:
            means = rgb[neutral].reshape(-1, 3).mean(axis=0)
        else:
            means = rgb.reshape(-1, 3).mean(axis=0)
        target = float(means.mean())
        gains = np.clip(target / np.maximum(means, 1e-4), 0.6, 1.6)
        balanced = np.clip(rgb * gains[None, None, :], 0.0, 1.0)
        out = rgb * (1.0 - strength) + balanced * strength
        return merge_alpha(out, alpha, was_u8)
