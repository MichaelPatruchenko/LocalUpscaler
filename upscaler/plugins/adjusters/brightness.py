"""Brightness adjustment."""
import numpy as np
from upscaler.plugins.base import BasePlugin


class BrightnessPlugin(BasePlugin):
    name = "Brightness"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "value": {
            "type": "integer",
            "minimum": -100,
            "maximum": 100,
            "default": 0,
            "ui": "slider",
        }
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        value = params.get("value", 0)
        return np.clip(image.astype(np.int16) + value, 0, 255).astype(np.uint8)
