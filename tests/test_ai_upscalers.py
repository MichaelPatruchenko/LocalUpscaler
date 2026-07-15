"""Tests for Real-ESRGAN AI upscaler plugin."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestRealESRGAN:
    def test_metadata(self):
        from upscaler.plugins.upscalers.real_esrgan import RealESRGANPlugin
        assert RealESRGANPlugin.name == "Real-ESRGAN"
        assert RealESRGANPlugin.category == "upscaler"
        assert 2 in RealESRGANPlugin.supported_scales
        assert 4 in RealESRGANPlugin.supported_scales
        assert RealESRGANPlugin.gpu_memory_mb > 0

    def test_params_schema(self):
        from upscaler.plugins.upscalers.real_esrgan import RealESRGANPlugin
        assert "model_variant" in RealESRGANPlugin.params_schema
        assert "tile_size" in RealESRGANPlugin.params_schema

    def test_tile_inference_helper(self):
        """Test the tiling helper produces correct output shape."""
        from upscaler.plugins.upscalers.real_esrgan import tile_process
        img = np.random.default_rng(42).random((64, 64, 3)).astype(np.float32)
        def mock_forward(tile):
            return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)
        result = tile_process(img, mock_forward, scale=2, tile_size=32, overlap=4)
        assert result.shape == (128, 128, 3)


class TestHATS:
    def test_metadata(self):
        from upscaler.plugins.upscalers.hat_s import HATSPlugin
        assert HATSPlugin.name == "HAT-S"
        assert HATSPlugin.supported_scales == [4]
        assert HATSPlugin.gpu_memory_mb > 0


class TestSwinIR:
    def test_metadata(self):
        from upscaler.plugins.upscalers.swinir import SwinIRPlugin
        assert SwinIRPlugin.name == "SwinIR"
        assert SwinIRPlugin.supported_scales == [2, 4]


class TestOmniSR:
    def test_metadata(self):
        from upscaler.plugins.upscalers.omnisr import OmniSRPlugin
        assert OmniSRPlugin.name == "OmniSR"
        assert OmniSRPlugin.supported_scales == [2, 4]


class TestDAT:
    def test_metadata(self):
        from upscaler.plugins.upscalers.dat import DATPlugin
        assert DATPlugin.name == "DAT"
        assert DATPlugin.supported_scales == [2, 4]
