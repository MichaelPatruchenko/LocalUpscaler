"""Auto contrast: histogram stretching per channel."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class AutoContrastPlugin(BasePlugin):
    name = "Auto Contrast"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "clip_percent": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 5.0,
            "default": 1.0,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        clip = params.get("clip_percent", 1.0)
        result = image.copy()
        for c in range(image.shape[2]) if image.ndim == 3 else [None]:
            ch = image[:, :, c] if c is not None else image
            total = ch.size
            clip_count = int(total * clip / 100.0)
            hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
            cumsum = np.cumsum(hist)
            low = np.searchsorted(cumsum, clip_count)
            high = np.searchsorted(cumsum, total - clip_count)
            high = max(high, low + 1)
            scale = 255.0 / (high - low)
            if c is not None:
                result[:, :, c] = np.clip(
                    (ch.astype(np.float32) - low) * scale, 0, 255
                ).astype(np.uint8)
            else:
                result = np.clip(
                    (ch.astype(np.float32) - low) * scale, 0, 255
                ).astype(np.uint8)
        return result
