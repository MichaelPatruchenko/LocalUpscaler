"""Lanczos interpolation upscaler."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class LanczosPlugin(BasePlugin):
    name = "Lanczos"
    category = "upscaler"
    supported_scales = [2, 4, 8, 16]
    gpu_memory_mb = 0
    params_schema = {
        "scale": {"type": "integer", "minimum": 2, "maximum": 16, "default": 2, "ui": "dropdown"}
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        scale = params.get("scale", 2)
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
