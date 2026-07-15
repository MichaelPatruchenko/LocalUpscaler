"""BM3D denoiser."""
import numpy as np
from upscaler.plugins.base import BasePlugin


class BM3DPlugin(BasePlugin):
    name = "BM3D"
    category = "denoiser"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "sigma": {
            "type": "number",
            "minimum": 1,
            "maximum": 75,
            "default": 25,
            "ui": "slider",
        },
        "fast": {
            "type": "boolean",
            "default": True,
            "ui": "checkbox",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        import bm3d as bm3d_lib

        sigma = params.get("sigma", 25) / 255.0
        original_dtype = image.dtype
        if original_dtype == np.uint8:
            img_f = image.astype(np.float64) / 255.0
        elif original_dtype == np.uint16:
            img_f = image.astype(np.float64) / 65535.0
        else:
            img_f = image.astype(np.float64)
        # Low-complexity profile ~halves runtime while keeping BM3D's character.
        profile = bm3d_lib.BM3DProfileLC() if params.get("fast", True) else "np"
        result = bm3d_lib.bm3d(img_f, sigma, profile=profile)
        if original_dtype == np.uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            return np.clip(result * 65535, 0, 65535).astype(np.uint16)
        return result.astype(np.float32)
