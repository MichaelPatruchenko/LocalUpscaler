"""Directional Cubic Convolution Interpolation (DCCI) — 2x upscaler.
Adapts interpolation direction based on local edge orientation."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class DCCIPlugin(BasePlugin):
    name = "DCCI"
    category = "upscaler"
    supported_scales = [2]
    gpu_memory_mb = 0
    params_schema = {
        "edge_threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.3,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        threshold = params.get("edge_threshold", 0.3)
        if image.ndim == 3:
            channels = [self._dcci_channel(image[:, :, c], threshold) for c in range(image.shape[2])]
            return np.stack(channels, axis=-1)
        return self._dcci_channel(image, threshold)

    def _dcci_channel(self, channel: np.ndarray, threshold: float) -> np.ndarray:
        h, w = channel.shape
        out = np.zeros((h * 2, w * 2), dtype=np.float64)
        img = channel.astype(np.float64)
        out[::2, ::2] = img
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        for y in range(h - 1):
            for x in range(w - 1):
                d1 = abs(img[y, x] - img[y + 1, x + 1])
                d2 = abs(img[y + 1, x] - img[y, x + 1])
                if d1 < d2 * (1 - threshold):
                    val = (img[y, x] + img[y + 1, x + 1]) / 2
                elif d2 < d1 * (1 - threshold):
                    val = (img[y + 1, x] + img[y, x + 1]) / 2
                else:
                    val = (img[y, x] + img[y + 1, x + 1] + img[y + 1, x] + img[y, x + 1]) / 4
                out[2 * y + 1, 2 * x + 1] = val
        for y in range(0, h * 2):
            for x in range(1, (w - 1) * 2, 2):
                if y % 2 == 0:
                    out[y, x] = (out[y, x - 1] + out[y, x + 1]) / 2
                else:
                    neighbors = []
                    if x > 0:
                        neighbors.append(out[y, x - 1])
                    if x + 1 < w * 2:
                        neighbors.append(out[y, x + 1])
                    if y > 0:
                        neighbors.append(out[y - 1, x])
                    if y + 1 < h * 2:
                        neighbors.append(out[y + 1, x])
                    out[y, x] = np.mean(neighbors) if neighbors else 0
        for y in range(1, (h - 1) * 2, 2):
            for x in range(0, w * 2, 2):
                neighbors = []
                if y > 0:
                    neighbors.append(out[y - 1, x])
                if y + 1 < h * 2:
                    neighbors.append(out[y + 1, x])
                out[y, x] = np.mean(neighbors) if neighbors else 0
        return np.clip(out, 0, 255 if channel.dtype == np.uint8 else 65535).astype(channel.dtype)
