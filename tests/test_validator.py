"""Tests for ResultValidator."""
import numpy as np
import pytest


class TestResultValidator:
    def test_validate_returns_dict(self, sample_rgb_uint8):
        from upscaler.engine.validator import ResultValidator
        v = ResultValidator()
        result = v.validate(sample_rgb_uint8, sample_rgb_uint8)
        assert "brisque" in result
        assert "artifacts" in result
        assert "histogram_similarity" in result

    def test_artifact_keys(self, sample_rgb_uint8):
        from upscaler.engine.validator import ResultValidator
        v = ResultValidator()
        result = v.validate(sample_rgb_uint8, sample_rgb_uint8)
        assert "banding" in result["artifacts"]
        assert "halos" in result["artifacts"]
        assert "ringing" in result["artifacts"]
