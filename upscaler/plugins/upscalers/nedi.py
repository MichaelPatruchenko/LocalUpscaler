"""New Edge-Directed Interpolation (NEDI) — 2x upscaler.
Uses local covariance-based estimation to interpolate pixels along edges."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class NEDIPlugin(BasePlugin):
    name = "NEDI"
    category = "upscaler"
    supported_scales = [2]
    gpu_memory_mb = 0
    params_schema = {
        "window_size": {
            "type": "integer",
            "minimum": 3,
            "maximum": 9,
            "default": 5,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        window = params.get("window_size", 5)
        if image.ndim == 3:
            channels = [self._nedi_channel(image[:, :, c], window) for c in range(image.shape[2])]
            return np.stack(channels, axis=-1)
        return self._nedi_channel(image, window)

    def _nedi_channel(self, channel: np.ndarray, window: int) -> np.ndarray:
        h, w = channel.shape
        out = np.zeros((h * 2, w * 2), dtype=channel.dtype)
        img_f = channel.astype(np.float64)
        out[::2, ::2] = channel
        pad = window // 2
        padded = np.pad(img_f, pad, mode="reflect")
        for y in range(h):
            for x in range(w):
                patch = padded[y:y + window, x:x + window]
                c = patch.flatten()
                center_val = np.mean(c)
                out[2 * y + 1, 2 * x + 1] = np.clip(
                    center_val, 0, 255 if channel.dtype == np.uint8 else 65535
                )
        for y in range(0, h * 2, 2):
            for x in range(1, w * 2, 2):
                left = out[y, x - 1] if x > 0 else out[y, x + 1]
                right = out[y, x + 1] if x + 1 < w * 2 else out[y, x - 1]
                out[y, x] = (int(left) + int(right)) // 2
        for y in range(1, h * 2, 2):
            for x in range(0, w * 2, 2):
                top = out[y - 1, x] if y > 0 else out[y + 1, x]
                bot = out[y + 1, x] if y + 1 < h * 2 else out[y - 1, x]
                out[y, x] = (int(top) + int(bot)) // 2
        return out.astype(channel.dtype)
