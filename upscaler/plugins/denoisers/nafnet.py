"""NAFNet: Nonlinear Activation Free Network for Image Restoration."""
import logging
import numpy as np
import torch
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.models.loader import load_model_from_file
from upscaler.engine.gpu_utils import safe_tile_process
from upscaler.config import MODELS_DIR

log = logging.getLogger(__name__)

NAFNET_VARIANTS = {
    "SIDD": "NAFNet-SIDD",
    "GoPro": "NAFNet-GoPro",
}


class NAFNetPlugin(BasePlugin):
    name = "NAFNet"
    category = "denoiser"
    supported_scales = []
    gpu_memory_mb = 250
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5, "ui": "slider"},
        "variant": {
            "type": "combo",
            "options": ["SIDD", "GoPro"],
            "default": "SIDD",
            "labels": {"SIDD": "SIDD (шумоподавление)", "GoPro": "GoPro (удаление размытия)"},
            "ui": "combo",
        },
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
        variant = params.get("variant", "SIDD")

        registry_name = NAFNET_VARIANTS.get(variant, "NAFNet-SIDD")

        if not self.model_manager.is_downloaded(registry_name):
            self.model_manager.download(registry_name)

        path = self.model_manager.get_model_path(registry_name)
        log.info(f"Loading NAFNet variant={variant} from {path}")
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
