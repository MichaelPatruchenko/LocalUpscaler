"""Wavelet-based denoiser (PyWavelets)."""
import numpy as np
from upscaler.plugins.base import BasePlugin


class WaveletPlugin(BasePlugin):
    name = "Wavelet"
    category = "denoiser"
    supported_scales = []
    gpu_memory_mb = 0
    params_schema = {
        "strength": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "default": 0.5,
            "ui": "slider",
        },
        "wavelet": {
            "type": "string",
            "enum": ["db1", "db2", "sym4", "coif1"],
            "default": "db2",
            "ui": "dropdown",
        },
        "level": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "default": 3,
            "ui": "slider",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 0.5)
        wavelet = params.get("wavelet", "db2")
        level = params.get("level", 3)
        original_dtype = image.dtype
        if image.ndim == 3:
            channels = []
            for c in range(image.shape[2]):
                channels.append(
                    self._denoise_channel(
                        image[:, :, c].astype(np.float64), wavelet, level, strength
                    )
                )
            result = np.stack(channels, axis=-1)
        else:
            result = self._denoise_channel(
                image.astype(np.float64), wavelet, level, strength
            )
        if original_dtype == np.uint8:
            return np.clip(result, 0, 255).astype(np.uint8)
        elif original_dtype == np.uint16:
            return np.clip(result, 0, 65535).astype(np.uint16)
        return result.astype(np.float32)

    def _denoise_channel(
        self, channel: np.ndarray, wavelet: str, level: int, strength: float
    ) -> np.ndarray:
        import pywt  # optional dependency, imported lazily
        max_level = pywt.dwt_max_level(min(channel.shape), pywt.Wavelet(wavelet).dec_len)
        level = min(level, max_level)
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail[0])) / 0.6745
        threshold = sigma * strength * 3
        new_coeffs = [coeffs[0]]
        for detail_tuple in coeffs[1:]:
            new_detail = tuple(
                pywt.threshold(d, threshold, mode="soft") for d in detail_tuple
            )
            new_coeffs.append(new_detail)
        return pywt.waverec2(new_coeffs, wavelet)[: channel.shape[0], : channel.shape[1]]
