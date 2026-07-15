"""Non-Local Means denoiser (OpenCV)."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class NLMeansPlugin(BasePlugin):
    name = "NL-Means"
    category = "denoiser"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {
            "type": "number",
            "minimum": 1,
            "maximum": 40,
            "default": 10,
            "ui": "slider",
        },
        "h": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "default": 10,
            "ui": "slider",
        },
        "template_window": {
            "type": "integer",
            "minimum": 3,
            "maximum": 21,
            "default": 7,
            "ui": "slider",
        },
        "search_window": {
            "type": "integer",
            "minimum": 3,
            "maximum": 50,
            "default": 21,
            "ui": "slider",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        h = params.get("h", int(params.get("strength", 10)))
        template_window = params.get("template_window", 7)
        search_window = params.get("search_window", 21)
        if image.ndim == 3:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h, h, template_window, search_window)
            return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
