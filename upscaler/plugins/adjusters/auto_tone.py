"""Auto tone: CLAHE-based adaptive tone mapping."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class AutoTonePlugin(BasePlugin):
    name = "Auto Tone"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.4,
            "ui": "slider",
        },
        "clip_limit": {
            "type": "number",
            "minimum": 0.5,
            "maximum": 5.0,
            "default": 1.0,
            "ui": "slider",
        },
        "grid_size": {
            "type": "integer",
            "minimum": 2,
            "maximum": 16,
            "default": 8,
            "ui": "slider",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 0.4)
        clip_limit = params.get("clip_limit", 1.0)
        grid = params.get("grid_size", 8)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
        lab_adjusted = lab.copy()
        lab_adjusted[:, :, 0] = clahe.apply(lab[:, :, 0])
        adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)
        # Blend with original to control intensity
        result = cv2.addWeighted(image, 1.0 - strength, adjusted, strength, 0)
        return result
