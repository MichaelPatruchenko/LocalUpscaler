import numpy as np
import pytest


class TestBRISQUE:
    def test_returns_float(self, sample_rgb_uint8):
        from upscaler.utils.metrics import compute_brisque
        score = compute_brisque(sample_rgb_uint8)
        assert isinstance(score, float)

    def test_score_in_range(self, sample_rgb_uint8):
        from upscaler.utils.metrics import compute_brisque
        score = compute_brisque(sample_rgb_uint8)
        assert 0 <= score <= 100


class TestHistogramComparison:
    def test_identical_images(self, sample_rgb_uint8):
        from upscaler.utils.metrics import histogram_similarity
        score = histogram_similarity(sample_rgb_uint8, sample_rgb_uint8)
        assert score > 0.99

    def test_different_images(self, sample_rgb_uint8):
        from upscaler.utils.metrics import histogram_similarity
        other = 255 - sample_rgb_uint8
        score = histogram_similarity(sample_rgb_uint8, other)
        assert score < 0.5


class TestArtifactDetection:
    def test_no_artifacts_in_smooth(self):
        from upscaler.utils.metrics import detect_artifacts
        smooth = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = detect_artifacts(smooth)
        assert isinstance(result, dict)
        assert "banding" in result
        assert "halos" in result
        assert "ringing" in result
