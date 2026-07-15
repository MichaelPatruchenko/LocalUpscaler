"""Saturation adjustment via HSV."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class SaturationPlugin(BasePlugin):
    name = "Saturation"
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
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
