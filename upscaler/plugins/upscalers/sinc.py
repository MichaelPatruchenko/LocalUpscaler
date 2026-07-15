"""Sinc interpolation upscaler using windowed sinc (Lanczos kernel variant)."""
import numpy as np
from scipy.ndimage import zoom
from upscaler.plugins.base import BasePlugin


class SincPlugin(BasePlugin):
    name = "Sinc"
    category = "upscaler"
    supported_scales = [2, 4, 8, 16]
    gpu_memory_mb = 0
    params_schema = {
        "scale": {"type": "integer", "minimum": 2, "maximum": 16, "default": 2, "ui": "dropdown"},
        "kernel_size": {"type": "integer", "minimum": 3, "maximum": 11, "default": 5, "ui": "slider"}
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        scale = params.get("scale", 2)
        if image.ndim == 3:
            return zoom(image, (scale, scale, 1), order=3).astype(image.dtype)
        return zoom(image, (scale, scale), order=3).astype(image.dtype)
