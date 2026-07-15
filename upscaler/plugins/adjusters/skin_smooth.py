"""Skin Smooth: frequency separation только в маске кожи внутри зон лиц."""
import logging

import cv2
import numpy as np

from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import split_alpha, merge_alpha

log = logging.getLogger(__name__)


class SkinSmoothPlugin(BasePlugin):
    name = "Skin Smooth"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                     "default": 0.5, "ui": "slider"},
        "radius": {"type": "integer", "minimum": 3, "maximum": 30,
                   "default": 10, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def _detect_faces(self, rgb_u8: np.ndarray) -> list:
        """Отделено для мокаемости; graceful при отсутствии модели."""
        from upscaler.plugins.face import facedet
        from upscaler.models.manager import ModelManager
        from upscaler.config import MODELS_DIR
        mm = ModelManager(MODELS_DIR)
        path = mm.get_model_path("YuNet")
        if not path.exists():
            mm.download("YuNet")
        return facedet.detect_faces(rgb_u8, str(path))

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = float(params.get("strength", 0.5))
        radius = int(params.get("radius", 10))
        if strength <= 0.0:
            return image
        rgb, alpha, was_u8 = split_alpha(image)
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
        try:
            faces = self._detect_faces(rgb_u8)
        except Exception as exc:
            log.warning("Skin Smooth: детекция лиц недоступна (%s); пропуск",
                        exc)
            return image
        if not faces:
            return image

        h, w = rgb.shape[:2]
        # Маска кожи: YCrCb-диапазон внутри расширенных bbox лиц.
        ycrcb = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)
        skin = ((ycrcb[..., 0] > 40)
                & (ycrcb[..., 1] >= 135) & (ycrcb[..., 1] <= 180)
                & (ycrcb[..., 2] >= 85) & (ycrcb[..., 2] <= 135))
        region = np.zeros((h, w), bool)
        for face in faces:
            x, y, fw, fh = face.bbox
            dx, dy = int(fw * 0.3), int(fh * 0.3)
            x0, y0 = max(0, int(x) - dx), max(0, int(y) - dy)
            x1 = min(w, int(x + fw) + dx)
            y1 = min(h, int(y + fh) + dy)
            region[y0:y1, x0:x1] = True
        mask = (skin & region).astype(np.float32)
        if mask.sum() < 16:
            return image
        mask = cv2.GaussianBlur(mask, (0, 0), 5.0)[..., None]

        sigma = max(radius / 2.0, 1.0)
        low = cv2.GaussianBlur(rgb, (0, 0), sigma)
        high = rgb - low
        # Сглаживаем высокие частоты (поры/шум) только в маске кожи.
        out = low + high * (1.0 - strength * mask)
        return merge_alpha(out, alpha, was_u8)
