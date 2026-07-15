"""Base plugin interface and category enum."""

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class PluginCategory(str, Enum):
    """Enum of plugin categories. Inherits from str so values compare equal to strings."""

    UPSCALER = "upscaler"
    DENOISER = "denoiser"
    ADJUSTER = "adjuster"
    DEBLUR = "deblur"
    ICEDIT = "icedit"
    FACE = "face"


class BasePlugin(ABC):
    """Abstract base for all processing plugins."""

    name: str = ""
    category: str = ""
    supported_scales: list = []
    gpu_memory_mb: int = 0
    params_schema: dict = {}
    supports_video: bool = False

    @abstractmethod
    def initialize(self, device: str) -> None:
        """Load model weights or prepare resources."""

    @abstractmethod
    def process(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Process image and return result. Image is RGB float32 [0,1] or uint8."""

    def cleanup(self) -> None:
        """Release resources (GPU memory, etc.)."""
        pass
