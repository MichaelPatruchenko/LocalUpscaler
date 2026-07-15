"""Unsharp mask sharpening."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class SharpnessPlugin(BasePlugin):
    name = "Sharpness"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "amount": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 3.0,
            "default": 0.5,
            "ui": "slider",
        },
        "radius": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 5.0,
            "default": 1.0,
            "ui": "slider",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        amount = params.get("amount", 0.5)
        radius = params.get("radius", 1.0)
        blurred = cv2.GaussianBlur(image, (0, 0), radius)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
