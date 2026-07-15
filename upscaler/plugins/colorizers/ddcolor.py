"""DDColor: photo colorization using dual decoders."""
import logging
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.engine.gpu_utils import safe_forward
from upscaler.config import MODELS_DIR

log = logging.getLogger(__name__)

DDCOLOR_VARIANTS = {
    "artistic": "DDColor-artistic",
    "modelscope": "DDColor-modelscope",
}


class DDColorPlugin(BasePlugin):
    name = "DDColor"
    category = "colorizer"
    supported_scales = []
    gpu_memory_mb = 900
    supports_video = False
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0, "ui": "slider"},
        "variant": {
            "type": "combo",
            "options": ["artistic", "modelscope"],
            "default": "artistic",
            "labels": {"artistic": "Artistic (яркие цвета)", "modelscope": "ModelScope (реалистичные)"},
            "ui": "combo",
        },
        "input_size": {"type": "integer", "minimum": 256, "maximum": 1024, "default": 512},
    }

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.model_manager = ModelManager(MODELS_DIR)

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)

    def _load_model(self, variant: str, input_size: int):
        """Load DDColor model using bundled architecture."""
        from upscaler.models.arch.ddcolor_arch import DDColor

        registry_name = DDCOLOR_VARIANTS.get(variant, "DDColor-artistic")

        if not self.model_manager.is_downloaded(registry_name):
            log.info(f"Downloading DDColor variant: {variant}")
            self.model_manager.download(registry_name)

        path = self.model_manager.get_model_path(registry_name)
        log.info(f"Loading DDColor variant={variant} from {path}")

        # Build model architecture
        model = DDColor(
            encoder_name="convnext-l",
            decoder_name="MultiScaleColorDecoder",
            input_size=[input_size, input_size],
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        )

        # Load weights
        state_dict = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "params" in state_dict:
            state_dict = state_dict["params"]
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        return model

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 1.0)
        input_size = params.get("input_size", 512)
        variant = params.get("variant", "artistic")

        if self.model is None:
            self.model = self._load_model(variant, input_size)

        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_f = image.astype(np.float32) / 65535.0
        else:
            img_f = image.astype(np.float32)

        h, w = img_f.shape[:2]

        # Convert RGB to BGR for OpenCV LAB conversion
        img_bgr = cv2.cvtColor((img_f * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_bgr_f = img_bgr.astype(np.float32) / 255.0

        # Extract original L channel at full resolution
        orig_l = cv2.cvtColor(img_bgr_f, cv2.COLOR_BGR2Lab)[:, :, :1]

        # Resize and create grayscale input
        img_resized = cv2.resize(img_bgr_f, (input_size, input_size))
        img_l = cv2.cvtColor(img_resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate([img_l, np.zeros_like(img_l), np.zeros_like(img_l)], axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_Lab2RGB)

        # To tensor
        tensor_input = (
            torch.from_numpy(img_gray_rgb.transpose((2, 0, 1)))
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        output_ab = safe_forward(tensor_input, self.model, self.device)

        # Resize ab channels to original resolution
        output_ab_resized = (
            F.interpolate(output_ab, size=(h, w), mode="bilinear", align_corners=False)[0]
            .numpy()
            .transpose(1, 2, 0)
        )

        # Combine with original L channel
        output_lab = np.concatenate([orig_l, output_ab_resized], axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_Lab2BGR)
        output_rgb = cv2.cvtColor((output_bgr * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        colorized = output_rgb.astype(np.float32) / 255.0

        # Blend with original
        result = img_f * (1 - strength) + colorized * strength

        if image.dtype == np.uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return np.clip(result * 65535, 0, 65535).astype(np.uint16)
        return result.astype(np.float32)

    def cleanup(self) -> None:
        self.model = None
        self.model_manager.unload_current()
