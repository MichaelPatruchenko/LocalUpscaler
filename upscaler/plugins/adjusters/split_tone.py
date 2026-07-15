"""Split Toning: тонирование теней и светов разными оттенками."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin
from upscaler.plugins.adjusters.common import (
    split_alpha, merge_alpha, luminance, scale_by_luminance,
)


def _hue_to_rgb(hue_deg: float) -> np.ndarray:
    hsv = np.array([[[hue_deg / 2.0, 255, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return rgb.astype(np.float32) / 255.0


class SplitTonePlugin(BasePlugin):
    name = "Split Toning"
    category = "adjuster"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "shadow_hue": {"type": "number", "minimum": 0.0, "maximum": 360.0,
                       "default": 215.0, "ui": "slider"},
        "highlight_hue": {"type": "number", "minimum": 0.0, "maximum": 360.0,
                          "default": 45.0, "ui": "slider"},
        "saturation": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                       "default": 0.25, "ui": "slider"},
        "balance": {"type": "number", "minimum": -1.0, "maximum": 1.0,
                    "default": 0.0, "ui": "slider"},
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        saturation = float(params.get("saturation", 0.25))
        if saturation <= 0.0:
            return image
        sh_rgb = _hue_to_rgb(float(params.get("shadow_hue", 215.0)) % 360.0)
        hi_rgb = _hue_to_rgb(float(params.get("highlight_hue", 45.0)) % 360.0)
        balance = float(params.get("balance", 0.0))
        rgb, alpha, was_u8 = split_alpha(image)
        lum = luminance(rgb)
        w_hi = np.clip((lum - 0.5 + balance * 0.25) / 0.5, 0.0, 1.0)[..., None]
        w_sh = 1.0 - w_hi
        a = saturation * 0.3
        tinted = (rgb * (1.0 - a)
                  + (w_sh * sh_rgb[None, None, :]
                     + w_hi * hi_rgb[None, None, :]) * a)
        # Вернуть исходную светимость (тонируем цвет, не яркость).
        out = scale_by_luminance(tinted, luminance(tinted), lum)
        return merge_alpha(out, alpha, was_u8)
