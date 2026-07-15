"""SCUNet: Practical Blind Denoising via Swin-Conv-UNet."""
import numpy as np
import torch
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.models.loader import load_model_from_file
from upscaler.engine.gpu_utils import safe_tile_process
from upscaler.config import MODELS_DIR


class SCUNetPlugin(BasePlugin):
    name = "SCUNet"
    category = "denoiser"
    supported_scales = []
    gpu_memory_mb = 300
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5, "ui": "slider"},
        "tile_size": {"type": "integer", "minimum": 128, "maximum": 1024, "default": 512, "ui": "slider"},
    }

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.model_manager = ModelManager(MODELS_DIR)

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 0.5)
        tile_size = params.get("tile_size", 512)
        if not self.model_manager.is_downloaded("SCUNet"):
            self.model_manager.download("SCUNet")

        path = self.model_manager.get_model_path("SCUNet")
        self.model = load_model_from_file(path, self.device)

        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        else:
            img_f = image.astype(np.float32)

        denoised = safe_tile_process(
            img_f,
            scale=1,
            device=self.device,
            model=self.model,
            tile_size_hint=tile_size,
            overlap=32,
            model_memory_mb=self.gpu_memory_mb,
        )
        result = img_f * (1 - strength) + denoised * strength
        if image.dtype == np.uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        return result.astype(np.float32)

    def cleanup(self) -> None:
        self.model = None
        self.model_manager.unload_current()
