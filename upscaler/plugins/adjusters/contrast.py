"""Contrast adjustment via linear scaling around mean."""
import numpy as np
from upscaler.plugins.base import BasePlugin


class ContrastPlugin(BasePlugin):
    name = "Contrast"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "factor": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 3.0,
            "default": 1.0,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        factor = params.get("factor", 1.0)
        mean = image.mean()
        result = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(result, 0, 255).astype(np.uint8)
