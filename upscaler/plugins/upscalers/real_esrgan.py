"""Real-ESRGAN upscaler plugin with tile-based inference and OOM retry."""

import numpy as np
import torch
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.models.loader import load_model_from_file
from upscaler.engine.gpu_utils import safe_tile_process
from upscaler.config import MODELS_DIR


def tile_process(
    image: np.ndarray,
    forward_fn,
    scale: int,
    tile_size: int = 512,
    overlap: int = 32,
) -> np.ndarray:
    """Process image in tiles with overlap and blending.

    Auto-detects actual model scale from first tile output, so the caller's
    ``scale`` hint is used only as a fallback.
    """
    h, w, c = image.shape
    actual_scale = scale
    output = None
    weights = None
    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)
            tile = image[y_start:y_end, x_start:x_end]
            result_tile = forward_fn(tile)

            # Auto-detect actual scale from first tile
            if output is None:
                actual_scale = round(result_tile.shape[0] / tile.shape[0])
                if actual_scale < 1:
                    actual_scale = 1
                out_h, out_w = h * actual_scale, w * actual_scale
                output = np.zeros((out_h, out_w, c), dtype=np.float32)
                weights = np.zeros((out_h, out_w, c), dtype=np.float32)

            oy = y_start * actual_scale
            ox = x_start * actual_scale
            oh = result_tile.shape[0]
            ow = result_tile.shape[1]
            # Clip to output bounds (safety)
            oh_clip = min(oh, output.shape[0] - oy)
            ow_clip = min(ow, output.shape[1] - ox)
            wy = _blend_weight(oh, overlap * actual_scale)
            wx = _blend_weight(ow, overlap * actual_scale)
            w_tile = (wy[:, None, None] * wx[None, :, None]).astype(np.float32)
            output[oy:oy + oh_clip, ox:ox + ow_clip] += result_tile[:oh_clip, :ow_clip] * w_tile[:oh_clip, :ow_clip]
            weights[oy:oy + oh_clip, ox:ox + ow_clip] += w_tile[:oh_clip, :ow_clip]

    if output is None:
        return image.copy()
    weights = np.maximum(weights, 1e-8)
    return output / weights


def _blend_weight(size: int, overlap: int) -> np.ndarray:
    w = np.ones(size, dtype=np.float32)
    if overlap > 0 and overlap < size:
        ramp = np.linspace(0, np.pi / 2, overlap)
        w[:overlap] = np.sin(ramp) ** 2
        w[-overlap:] = np.cos(ramp) ** 2
    return w


class RealESRGANPlugin(BasePlugin):
    name = "Real-ESRGAN"
    category = "upscaler"
    supported_scales = [2, 4]
    gpu_memory_mb = 500
    params_schema = {
        "model_variant": {"type": "string", "enum": ["x2plus", "x4plus"], "default": "x4plus", "ui": "dropdown"},
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
        model_name = f"Real-ESRGAN-x{scale}"
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

