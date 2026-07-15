"""Tests for traditional interpolation upscalers: Lanczos, Bicubic, Sinc."""
import numpy as np
import pytest


@pytest.fixture
def small_image():
    return np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)


class TestLanczos:
    def test_upscale_2x(self, small_image):
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        p = LanczosPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 2})
        assert result.shape == (64, 64, 3)

    def test_upscale_4x(self, small_image):
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        p = LanczosPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 4})
        assert result.shape == (128, 128, 3)

    def test_metadata(self):
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        assert LanczosPlugin.name == "Lanczos"
        assert LanczosPlugin.category == "upscaler"
        assert LanczosPlugin.gpu_memory_mb == 0


class TestBicubic:
    def test_upscale_2x(self, small_image):
        from upscaler.plugins.upscalers.bicubic import BicubicPlugin
        p = BicubicPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 2})
        assert result.shape == (64, 64, 3)

    def test_metadata(self):
        from upscaler.plugins.upscalers.bicubic import BicubicPlugin
        assert BicubicPlugin.name == "Bicubic"
        assert BicubicPlugin.category == "upscaler"


class TestSinc:
    def test_upscale_2x(self, small_image):
        from upscaler.plugins.upscalers.sinc import SincPlugin
        p = SincPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 2})
        assert result.shape == (64, 64, 3)

    def test_metadata(self):
        from upscaler.plugins.upscalers.sinc import SincPlugin
        assert SincPlugin.name == "Sinc"
