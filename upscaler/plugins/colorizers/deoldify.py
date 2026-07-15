"""DeOldify: colorization for photos and videos."""
import logging
import numpy as np
import torch
import cv2
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.engine.gpu_utils import safe_forward
from upscaler.config import MODELS_DIR

log = logging.getLogger(__name__)

DEOLDIFY_VARIANTS = {
    "stable": "DeOldify-stable",
    "artistic": "DeOldify-artistic",
    "video": "DeOldify-video",
}

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DeOldifyPlugin(BasePlugin):
    name = "DeOldify"
    category = "colorizer"
    supported_scales = []
    gpu_memory_mb = 600
    supports_video = True
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0, "ui": "slider"},
        "variant": {
            "type": "combo",
            "options": ["stable", "artistic", "video"],
            "default": "stable",
            "labels": {
                "stable": "Stable (стабильная)",
                "artistic": "Artistic (художественная)",
                "video": "Video (для видео)",
            },
            "ui": "combo",
        },
        "render_factor": {"type": "integer", "minimum": 7, "maximum": 45, "default": 35},
    }

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.model_manager = ModelManager(MODELS_DIR)
        self._current_variant = None

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)

    def _load_model(self, variant: str):
        """Load DeOldify model using bundled architecture."""
        from upscaler.models.arch.deoldify_arch import build_deoldify_model

        registry_name = DEOLDIFY_VARIANTS.get(variant, "DeOldify-stable")

        if not self.model_manager.is_downloaded(registry_name):
            log.info(f"Downloading DeOldify variant: {variant}")
            self.model_manager.download(registry_name)

        path = self.model_manager.get_model_path(registry_name)
        log.info(f"Loading DeOldify variant={variant} from {path}")

        model = build_deoldify_model(variant)

        # Load weights - use weights_only=False for legacy pickle format
        state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        if isinstance(state_dict, dict) and "opt" in state_dict:
            # fastai checkpoint format
            state_dict = {k: v for k, v in state_dict.items() if k != "opt"}

        # Try to load state dict, allowing mismatches for inference
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            log.warning("Strict loading failed, trying non-strict")
            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device).eval()
        self._current_variant = variant
        return model

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 1.0)
        render_factor = params.get("render_factor", 35)
        variant = params.get("variant", "stable")

        if self.model is None or self._current_variant != variant:
            self.model = self._load_model(variant)

        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_f = image.astype(np.float32) / 65535.0
        else:
            img_f = image.astype(np.float32)

        h, w = img_f.shape[:2]

        # Convert to grayscale (like DeOldify: RGB → Gray → RGB)
        gray = cv2.cvtColor((img_f * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

        # Resize to render_factor * 16
        target_size = render_factor * 16
        resized = cv2.resize(gray_rgb, (target_size, target_size))

        # Normalize with ImageNet stats
        normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD

        # To tensor NCHW
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        output = safe_forward(tensor, self.model, self.device)

        # Denormalize output
        out_np = output.squeeze(0).numpy().transpose(1, 2, 0)
        out_denorm = out_np * IMAGENET_STD + IMAGENET_MEAN
        out_denorm = np.clip(out_denorm, 0, 1)

        # Resize to original dimensions
        colorized_resized = cv2.resize(out_denorm, (w, h))

        # YUV channel swap: keep original luminance, take model's chrominance
        orig_yuv = cv2.cvtColor((img_f * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
        color_yuv = cv2.cvtColor((colorized_resized * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2YUV)

        # Replace U and V channels
        merged_yuv = orig_yuv.copy()
        merged_yuv[:, :, 1] = color_yuv[:, :, 1]
        merged_yuv[:, :, 2] = color_yuv[:, :, 2]

        colorized = cv2.cvtColor(merged_yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0

        # Blend with original
        result = img_f * (1 - strength) + colorized * strength

        if image.dtype == np.uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return np.clip(result * 65535, 0, 65535).astype(np.uint16)
        return result.astype(np.float32)

    def process_video_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
        return self.process(frame, params)

    def cleanup(self) -> None:
        self.model = None
        self._current_variant = None
        self.model_manager.unload_current()
