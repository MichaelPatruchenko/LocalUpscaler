"""Bilateral filter denoiser (OpenCV)."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class BilateralPlugin(BasePlugin):
    name = "Bilateral"
    category = "denoiser"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "d": {"type": "integer", "minimum": 3, "maximum": 15, "default": 9, "ui": "slider"},
        "sigma_color": {
            "type": "number",
            "minimum": 10,
            "maximum": 200,
            "default": 75,
            "ui": "slider",
        },
        "sigma_space": {
            "type": "number",
            "minimum": 10,
            "maximum": 200,
            "default": 75,
            "ui": "slider",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        d = params.get("d", 9)
        sc = params.get("sigma_color", 75)
        ss = params.get("sigma_space", 75)
        return cv2.bilateralFilter(image, d, sc, ss)
