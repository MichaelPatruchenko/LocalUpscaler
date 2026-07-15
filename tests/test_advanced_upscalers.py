"""Tests for advanced traditional upscalers: NEDI, DCCI, EGGI/SIRE."""
import numpy as np
import pytest


@pytest.fixture
def small_image():
    return np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)


class TestNEDI:
    def test_upscale_2x(self, small_image):
        from upscaler.plugins.upscalers.nedi import NEDIPlugin
        p = NEDIPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 2})
        assert result.shape == (64, 64, 3)

    def test_only_supports_2x(self):
        from upscaler.plugins.upscalers.nedi import NEDIPlugin
        assert NEDIPlugin.supported_scales == [2]


class TestDCCI:
    def test_upscale_2x(self, small_image):
        from upscaler.plugins.upscalers.dcci import DCCIPlugin
        p = DCCIPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 2})
        assert result.shape == (64, 64, 3)

    def test_only_supports_2x(self):
        from upscaler.plugins.upscalers.dcci import DCCIPlugin
        assert DCCIPlugin.supported_scales == [2]


class TestEGGI:
    def test_upscale_2x(self, small_image):
        from upscaler.plugins.upscalers.eggi_sire import EGGISIREPlugin
        p = EGGISIREPlugin()
        p.initialize("cpu")
        result = p.process(small_image, {"scale": 2})
        assert result.shape == (64, 64, 3)

    def test_only_supports_2x(self):
        from upscaler.plugins.upscalers.eggi_sire import EGGISIREPlugin
        assert EGGISIREPlugin.supported_scales == [2]
