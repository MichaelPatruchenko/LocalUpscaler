# tests/test_adjusters.py
import numpy as np
import pytest

@pytest.fixture
def test_image():
    rng = np.random.default_rng(42)
    return rng.integers(50, 200, (64, 64, 3), dtype=np.uint8)

class TestAutoContrast:
    def test_output_shape(self, test_image):
        from upscaler.plugins.adjusters.auto_contrast import AutoContrastPlugin
        p = AutoContrastPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {})
        assert result.shape == test_image.shape

    def test_stretches_range(self, test_image):
        from upscaler.plugins.adjusters.auto_contrast import AutoContrastPlugin
        p = AutoContrastPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {"clip_percent": 1.0})
        assert result.max() >= test_image.max()
        assert result.min() <= test_image.min()

class TestAutoTone:
    def test_output_shape(self, test_image):
        from upscaler.plugins.adjusters.auto_tone import AutoTonePlugin
        p = AutoTonePlugin()
        p.initialize("cpu")
        result = p.process(test_image, {})
        assert result.shape == test_image.shape

class TestAutoColor:
    def test_output_shape(self, test_image):
        from upscaler.plugins.adjusters.auto_color import AutoColorPlugin
        p = AutoColorPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {})
        assert result.shape == test_image.shape

class TestBrightness:
    def test_increase(self, test_image):
        from upscaler.plugins.adjusters.brightness import BrightnessPlugin
        p = BrightnessPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {"value": 30})
        assert result.mean() > test_image.mean()

class TestContrast:
    def test_output_shape(self, test_image):
        from upscaler.plugins.adjusters.contrast import ContrastPlugin
        p = ContrastPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {"factor": 1.5})
        assert result.shape == test_image.shape

class TestSaturation:
    def test_output_shape(self, test_image):
        from upscaler.plugins.adjusters.saturation import SaturationPlugin
        p = SaturationPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {"factor": 1.5})
        assert result.shape == test_image.shape

class TestSharpness:
    def test_output_shape(self, test_image):
        from upscaler.plugins.adjusters.sharpness import SharpnessPlugin
        p = SharpnessPlugin()
        p.initialize("cpu")
        result = p.process(test_image, {"amount": 0.5})
        assert result.shape == test_image.shape
