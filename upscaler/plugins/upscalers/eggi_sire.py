"""Edge-Guided Gradient Interpolation / SIRE — 2x upscaler.
Uses gradient-guided interpolation with robust estimation."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class EGGISIREPlugin(BasePlugin):
    name = "EGGI/SIRE"
    category = "upscaler"
    supported_scales = [2]
    gpu_memory_mb = 0
    params_schema = {
        "robustness": {
            "type": "number",
            "minimum": 0.1,
            "maximum": 2.0,
            "default": 1.0,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        robustness = params.get("robustness", 1.0)
        if image.ndim == 3:
            channels = [self._eggi_channel(image[:, :, c], robustness) for c in range(image.shape[2])]
            return np.stack(channels, axis=-1)
        return self._eggi_channel(image, robustness)

    def _eggi_channel(self, channel: np.ndarray, robustness: float) -> np.ndarray:
        h, w = channel.shape
        img = channel.astype(np.float64)
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        out = np.zeros((h * 2, w * 2), dtype=np.float64)
        out[::2, ::2] = img
        for y in range(h - 1):
            for x in range(w - 1):
                grad_mag = np.sqrt(gx[y, x] ** 2 + gy[y, x] ** 2)
                if grad_mag < 1e-6:
                    val = (img[y, x] + img[y + 1, x + 1] + img[y + 1, x] + img[y, x + 1]) / 4
                else:
                    angle = np.arctan2(gy[y, x], gx[y, x])
                    w1 = np.exp(-abs(np.cos(angle)) * robustness)
                    w2 = np.exp(-abs(np.sin(angle)) * robustness)
                    d1 = w1 * (img[y, x] + img[y + 1, x + 1])
                    d2 = w2 * (img[y + 1, x] + img[y, x + 1])
                    val = (d1 + d2) / (2 * (w1 + w2))
                out[2 * y + 1, 2 * x + 1] = val
        for y in range(0, h * 2, 2):
            for x in range(1, w * 2 - 1, 2):
                out[y, x] = (out[y, x - 1] + out[y, x + 1]) / 2
        for y in range(1, h * 2 - 1, 2):
            for x in range(0, w * 2, 2):
                out[y, x] = (out[y - 1, x] + out[y + 1, x]) / 2
        return np.clip(out, 0, 255 if channel.dtype == np.uint8 else 65535).astype(channel.dtype)
