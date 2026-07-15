"""Dehaze: dark channel prior (упрощённый) со смешиванием по силе."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import split_alpha, merge_alpha


class DehazePlugin(BasePlugin):
    name = "Dehaze"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.5, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.5))
        if strength <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark = cv2.erode(rgb.min(axis=2), kernel)
        # Атмосферный свет: средний цвет самых ярких 0.1% пикселей dark channel.
        flat = dark.ravel()
        n_top = max(int(flat.size * 0.001), 1)
        idx = np.argpartition(flat, -n_top)[-n_top:]
        ys, xs = np.unravel_index(idx, dark.shape)
        atmo = np.clip(rgb[ys, xs].mean(axis=0), 0.05, 0.95)
        # Карта передачи.
        norm_dark = cv2.erode((rgb / atmo[None, None, :]).min(axis=2), kernel)
        t = np.clip(1.0 - 0.95 * norm_dark, 0.1, 1.0)[..., None]
        restored = np.clip((rgb - atmo) / t + atmo, 0.0, 1.0)
        out = rgb * (1.0 - strength) + restored * strength
        return merge_alpha(out, alpha, was_u8)
