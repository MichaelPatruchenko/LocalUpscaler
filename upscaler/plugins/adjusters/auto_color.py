"""Auto color correction: gray world white balance."""
import numpy as np
from upscaler.plugins.base import BasePlugin


class AutoColorPlugin(BasePlugin):
    name = "Auto Color"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.8,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 0.8)
        img_f = image.astype(np.float32)
        avg = img_f.mean(axis=(0, 1))
        gray = avg.mean()
        scale = gray / (avg + 1e-6)
        corrected = img_f * scale * strength + img_f * (1 - strength)
        return np.clip(corrected, 0, 255).astype(np.uint8)
