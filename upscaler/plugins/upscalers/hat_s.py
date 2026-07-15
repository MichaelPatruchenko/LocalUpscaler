"""HAT-S: Hybrid Attention Transformer for Image Restoration (Small variant)."""
import numpy as np
import torch
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.models.loader import load_model_from_file
from upscaler.engine.gpu_utils import safe_tile_process
from upscaler.config import MODELS_DIR


class HATSPlugin(BasePlugin):
    name = "HAT-S"
    category = "upscaler"
    supported_scales = [4]
    gpu_memory_mb = 450
    params_schema = {
        "tile_size": {"type": "integer", "minimum": 128, "maximum": 1024, "default": 512, "ui": "slider"},
    }

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.model_manager = ModelManager(MODELS_DIR)

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        scale = params.get("scale", 4)
        tile_size = params.get("tile_size", 512)
        # Only a 4x model is available; pipeline handles multi-pass for other scales
        model_name = "HAT-S-x4"
        if not self.model_manager.is_downloaded(model_name):
            self.model_manager.download(model_name)

        path = self.model_manager.get_model_path(model_name)
        self.model = load_model_from_file(path, self.device)

        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_f = image.astype(np.float32) / 65535.0
        else:
            img_f = image.astype(np.float32)

        result = safe_tile_process(
            img_f,
            scale=scale,
            device=self.device,
            model=self.model,
            tile_size_hint=tile_size,
            overlap=32,
            model_memory_mb=self.gpu_memory_mb,
        )

        if image.dtype == np.uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return np.clip(result * 65535, 0, 65535).astype(np.uint16)
        return result

    def cleanup(self) -> None:
        self.model = None
        self.model_manager.unload_current()
