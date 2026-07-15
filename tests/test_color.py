import numpy as np
import pytest


class TestColorConversions:
    def test_to_linear_and_back(self, sample_rgb_uint8):
        from upscaler.utils.color import srgb_to_linear, linear_to_srgb
        linear = srgb_to_linear(sample_rgb_uint8)
        assert linear.dtype == np.float32
        assert linear.min() >= 0.0
        assert linear.max() <= 1.0
        back = linear_to_srgb(linear)
        assert back.dtype == np.uint8
        # Allow +-1 rounding error
        assert np.abs(back.astype(int) - sample_rgb_uint8.astype(int)).max() <= 1

    def test_normalize_to_float(self, sample_rgb_uint8):
        from upscaler.utils.color import normalize_to_float
        f = normalize_to_float(sample_rgb_uint8)
        assert f.dtype == np.float32
        assert f.max() <= 1.0
        assert f.min() >= 0.0

    def test_normalize_16bit(self):
        from upscaler.utils.color import normalize_to_float
        img = np.array([0, 32768, 65535], dtype=np.uint16).reshape(1, 3, 1)
        f = normalize_to_float(img)
        assert abs(f[0, 1, 0] - 0.5) < 0.01

    def test_float_already_normalized(self, sample_rgb_float32):
        from upscaler.utils.color import normalize_to_float
        f = normalize_to_float(sample_rgb_float32)
        np.testing.assert_array_equal(f, sample_rgb_float32)

    def test_denormalize_to_uint8(self, sample_rgb_float32):
        from upscaler.utils.color import denormalize_to_dtype
        result = denormalize_to_dtype(sample_rgb_float32, np.uint8)
        assert result.dtype == np.uint8
        assert result.max() <= 255

    def test_denormalize_to_uint16(self, sample_rgb_float32):
        from upscaler.utils.color import denormalize_to_dtype
        result = denormalize_to_dtype(sample_rgb_float32, np.uint16)
        assert result.dtype == np.uint16
        assert result.max() <= 65535
