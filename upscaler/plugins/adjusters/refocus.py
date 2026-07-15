"""Multi-scale edge-aware refocus (sharpening) in LAB color space."""
import cv2
import numpy as np
from upscaler.plugins.base import BasePlugin


class RefocusPlugin(BasePlugin):
    name = "Refocus"
    category = "adjuster"
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
        "fine_detail": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 2.0,
            "default": 0.8,
            "ui": "slider",
        },
        "medium_detail": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 2.0,
            "default": 0.5,
            "ui": "slider",
        },
        "coarse_detail": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 2.0,
            "default": 0.3,
            "ui": "slider",
        },
    }

    def initialize(self, device: str) -> None:
        pass

    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        strength = params.get("strength", 0.5)
        fine = params.get("fine_detail", 0.8)
        medium = params.get("medium_detail", 0.5)
        coarse = params.get("coarse_detail", 0.3)

        if strength <= 0:
            return image

        is_uint8 = image.dtype == np.uint8
        if is_uint8:
            work = image.astype(np.float32) / 255.0
        else:
            work = image.astype(np.float32)

        # Work in LAB to only sharpen luminance (avoids color fringing)
        lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]

        # Multi-scale unsharp: extract detail at 3 scales
        blur_fine = cv2.GaussianBlur(L, (0, 0), 0.5)
        blur_medium = cv2.GaussianBlur(L, (0, 0), 1.5)
        blur_coarse = cv2.GaussianBlur(L, (0, 0), 3.0)

        detail_fine = L - blur_fine
        detail_medium = blur_fine - blur_medium
        detail_coarse = blur_medium - blur_coarse

        # Edge mask: suppress sharpening in flat/noisy areas, boost on edges
        edges = cv2.Laplacian(blur_medium, cv2.CV_32F)
        edge_mask = np.abs(edges)
        edge_mask = edge_mask / (edge_mask.max() + 1e-8)
        edge_mask = np.clip(edge_mask * 3.0, 0, 1)
        # Smooth the mask to avoid harsh transitions
        edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 2.0)

        # Apply weighted detail layers with edge awareness
        enhanced_L = L + strength * edge_mask * (
            fine * detail_fine
            + medium * detail_medium
            + coarse * detail_coarse
        )

        lab[:, :, 0] = np.clip(enhanced_L, 0, 100)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        if is_uint8:
            return np.clip(result * 255, 0, 255).astype(np.uint8)
        return np.clip(result, 0, 1).astype(np.float32)
