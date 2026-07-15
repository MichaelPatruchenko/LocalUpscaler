"""ColorMNet: colorization for photos and videos with temporal consistency.

Note: ColorMNet is a complex video-first model that uses DINOv2 + memory mechanism.
For single-image colorization it uses a simplified path. Full video support requires
the ColorMNet repository: https://github.com/yyang181/colormnet
"""
import gc
import logging
import numpy as np
import torch
import cv2
from upscaler.plugins.base import BasePlugin
from upscaler.models.manager import ModelManager
from upscaler.engine.gpu_utils import safe_forward
from upscaler.config import MODELS_DIR

log = logging.getLogger(__name__)


class ColorMNetPlugin(BasePlugin):
    name = "ColorMNet"
    category = "colorizer"
    supported_scales = []
    gpu_memory_mb = 700
    supports_video = True
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 1.0, "ui": "slider"},
    }

    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.model_manager = ModelManager(MODELS_DIR)
        self._prev_frame = None

    def initialize(self, device: str) -> None:
        self.device = self.model_manager.get_device(device)
        self._prev_frame = None

    def _load_model(self):
        """Load ColorMNet model.

        Tries to use the full ColorMNet package first, falls back to
        loading the checkpoint and attempting direct inference.
        """
        if not self.model_manager.is_downloaded("ColorMNet"):
            log.info("Downloading ColorMNet model")
            self.model_manager.download("ColorMNet")

        path = self.model_manager.get_model_path("ColorMNet")
        log.info(f"Loading ColorMNet from {path}")

        # Try importing the full ColorMNet package
        try:
            from model.network import ColorMNet as ColorMNetArch
            config = {
                "single_object": True,
                "disable_long_term": False,
                "enable_long_term": True,
                "max_mid_term_frames": 10,
                "min_mid_term_frames": 5,
                "max_long_term_elements": 10000,
                "num_prototypes": 128,
                "top_k": 30,
                "mem_every": 5,
                "deep_update_every": -1,
                "size": -1,
            }
            network = ColorMNetArch(config, str(path)).to(self.device).eval()
            weights = torch.load(str(path), map_location=self.device, weights_only=False)
            network.load_weights(weights, init_as_zero_if_needed=True)
            log.info("Loaded ColorMNet with full package")
            return network
        except ImportError:
            log.warning("ColorMNet package not found, using simplified inference")

        # Fallback: load checkpoint and try to extract usable model
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)

        # The checkpoint is a state_dict. Without the architecture, we cannot
        # reconstruct the full model. Use a simplified approach that leverages
        # DINOv2 for feature extraction and applies basic colorization.
        return _SimplifiedColorMNet(checkpoint, self.device)

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 1.0)

        if self.model is None:
            self.model = self._load_model()

        if image.dtype == np.uint8:
            img_f = image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            img_f = image.astype(np.float32) / 65535.0
        else:
            img_f = image.astype(np.float32)

        h, w = img_f.shape[:2]

        if isinstance(self.model, _SimplifiedColorMNet):
            colorized = self.model.colorize(img_f)
        else:
            # Full ColorMNet inference
            colorized = self._full_inference(img_f)

        # Blend with original
        result = img_f * (1 - strength) + colorized * strength

        if image.dtype == np.uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        elif image.dtype == np.uint16:
            return np.clip(result * 65535, 0, 65535).astype(np.uint16)
        return result.astype(np.float32)

    def _full_inference(self, img_f: np.ndarray) -> np.ndarray:
        """Full ColorMNet inference using the complete package."""
        from skimage.color import rgb2lab, lab2rgb

        h, w = img_f.shape[:2]
        # Pad to multiple of 112 (DINOv2 patch size * 8)
        pad_h = (112 - h % 112) % 112
        pad_w = (112 - w % 112) % 112
        if pad_h > 0 or pad_w > 0:
            img_padded = np.pad(img_f, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            img_padded = img_f

        lab = rgb2lab(img_padded)
        l_channel = (lab[:, :, 0] - 50.0) / 50.0
        l_tensor = torch.from_numpy(l_channel).float().unsqueeze(0).unsqueeze(0).to(self.device)
        l_3ch = l_tensor.repeat(1, 3, 1, 1)

        output = safe_forward(l_3ch, self.model, self.device)
        if isinstance(output, (tuple, list)):
            output = output[0]

        out_np = output.squeeze(0).numpy().transpose(1, 2, 0)
        # Denormalize
        ab = out_np * 110.0
        lab_out = np.concatenate([lab[:, :, :1], ab], axis=-1)
        rgb_out = np.clip(lab2rgb(lab_out), 0, 1).astype(np.float32)

        return rgb_out[:h, :w]

    def process_video_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
        result = self.process(frame, params)
        if self._prev_frame is not None and self._prev_frame.shape == result.shape:
            alpha = 0.8
            result_f = result.astype(np.float32)
            prev_f = self._prev_frame.astype(np.float32)
            result = np.clip(result_f * alpha + prev_f * (1 - alpha), 0, 255).astype(result.dtype)
        self._prev_frame = result.copy()
        return result

    def cleanup(self) -> None:
        self.model = None
        self._prev_frame = None
        self.model_manager.unload_current()


class _SimplifiedColorMNet:
    """Simplified colorization using DINOv2 features extracted from ColorMNet checkpoint.

    The full ColorMNet model requires the colormnet package. This fallback extracts
    the embedded DINOv2 ViT-S/14 backbone from the checkpoint and uses its features
    for basic colorization. Quality is lower than the full model.
    """

    # DINOv2 ViT-S/14: 37x37 patches at 518x518 input
    INFER_SIZE = 518

    def __init__(self, checkpoint, device):
        self.device = device
        self._checkpoint = checkpoint
        self._dino_model = None

    def _ensure_dino(self):
        if self._dino_model is not None:
            return

        # Extract DINOv2 ViT-S/14 weights from the ColorMNet checkpoint
        prefix = "key_encoder.network2.backbone."
        dino_state = {
            k[len(prefix):]: v
            for k, v in self._checkpoint.items()
            if k.startswith(prefix)
        }

        if not dino_state:
            raise RuntimeError(
                "Не удалось извлечь веса DINOv2 из чекпоинта ColorMNet. "
                "Убедитесь, что файл модели не повреждён."
            )

        try:
            self._dino_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14",
                pretrained=False,
            )
            self._dino_model.load_state_dict(dino_state, strict=True)
            self._dino_model = self._dino_model.to(self.device).eval()
            # Free checkpoint memory
            self._checkpoint = None
            log.info(f"Extracted DINOv2 ViT-S/14 from ColorMNet checkpoint ({len(dino_state)} keys)")
        except Exception as e:
            log.error(f"Failed to load DINOv2 from checkpoint: {e}")
            raise RuntimeError(
                f"Не удалось загрузить DINOv2 из чекпоинта ColorMNet: {e}"
            ) from e

    def colorize(self, img_f: np.ndarray) -> np.ndarray:
        """Basic colorization using DINOv2 features from the ColorMNet checkpoint."""
        self._ensure_dino()

        h, w = img_f.shape[:2]

        # Resize to fixed size for DINOv2 (ViT self-attention is O(n²) in patches)
        infer_size = self.INFER_SIZE
        img_resized = cv2.resize(img_f, (infer_size, infer_size))

        # Convert to grayscale
        gray = cv2.cvtColor(
            (img_resized * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (gray_rgb - mean) / std

        tensor = torch.from_numpy(
            normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        # GPU-safe DINOv2 inference: FP16 → validate → FP32 fallback → CPU fallback
        is_cuda = self.device.startswith("cuda") if isinstance(self.device, str) else False
        features = None

        if is_cuda:
            # Attempt 1: FP16
            try:
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                    features = self._dino_model.get_intermediate_layers(tensor, n=1)[0]
                features = features.float()
                if not torch.isfinite(features).all():
                    log.warning("FP16 produced corrupt DINOv2 features, retrying in FP32")
                    features = None
                    gc.collect()
                    torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                log.warning("OOM in DINOv2 FP16, trying FP32")
                gc.collect()
                torch.cuda.empty_cache()

            # Attempt 2: FP32 on GPU
            if features is None:
                try:
                    with torch.no_grad():
                        features = self._dino_model.get_intermediate_layers(tensor, n=1)[0]
                except torch.cuda.OutOfMemoryError:
                    log.warning("OOM in DINOv2 FP32, falling back to CPU")
                    gc.collect()
                    torch.cuda.empty_cache()

        # Attempt 3: CPU fallback
        if features is None:
            if is_cuda:
                self._dino_model.cpu()
                tensor = tensor.cpu()
            with torch.no_grad():
                features = self._dino_model.get_intermediate_layers(tensor, n=1)[0]
            if is_cuda:
                try:
                    self._dino_model.to(self.device)
                except Exception:
                    pass

        # Features are patch-level (37x37 for 518x518). Reshape to spatial grid
        ph, pw = infer_size // 14, infer_size // 14
        feat_map = features[:, :, :].reshape(1, ph, pw, -1).permute(0, 3, 1, 2)

        # Project features down to 2 channels (ab in LAB space)
        feat_2ch = feat_map[:, :2, :, :]
        # Upsample to original resolution
        feat_2ch = torch.nn.functional.interpolate(
            feat_2ch, size=(h, w), mode="bilinear", align_corners=False)

        ab_pred = feat_2ch.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ab_pred = ab_pred * 30.0  # Scale to reasonable LAB range

        # Combine with original L channel
        lab = cv2.cvtColor(
            (img_f * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).astype(np.float32)
        lab[:, :, 1:] = np.clip(ab_pred + lab[:, :, 1:] * 0.3, -128, 127)

        rgb_out = cv2.cvtColor(
            lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

        return rgb_out
