"""End-to-end tests for the processing pipeline (no GUI)."""

import numpy as np
import pytest
from pathlib import Path


class TestEndToEndPipeline:
    def test_full_pipeline_lanczos_2x(self, tmp_path, sample_rgb_uint8):
        """Full pipeline: load -> configure -> process -> validate."""
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        from upscaler.plugins.adjusters.auto_contrast import AutoContrastPlugin
        from upscaler.plugins.adjusters.sharpness import SharpnessPlugin
        from upscaler.engine.pipeline import PipelineExecutor

        reg = PluginRegistry()
        reg.register(LanczosPlugin)
        reg.register(AutoContrastPlugin)
        reg.register(SharpnessPlugin)

        pe = PipelineExecutor(reg)
        config = {
            "scale": 2,
            "enhance_only": False,
            "upscale": {"plugin": "Lanczos"},
            "adjust": {"Auto Contrast": {}},
            "post": {"sharpen": 0.3},
        }
        meta = {"format": ".png", "bit_depth": 8, "icc_profile": None}

        result = pe.execute(sample_rgb_uint8, config, meta)
        assert result["image"].shape == (128, 128, 3)
        assert "brisque" in result["metrics"]
        assert "artifacts" in result["metrics"]

    def test_enhance_only_pipeline(self, sample_rgb_uint8):
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.adjusters.brightness import BrightnessPlugin
        from upscaler.plugins.denoisers.nl_means import NLMeansPlugin
        from upscaler.engine.pipeline import PipelineExecutor

        reg = PluginRegistry()
        reg.register(BrightnessPlugin)
        reg.register(NLMeansPlugin)

        pe = PipelineExecutor(reg)
        config = {
            "scale": 1,
            "enhance_only": True,
            "denoise": {"NL-Means": {"strength": 5}},
            "adjust": {"Brightness": {"value": 10}},
        }
        meta = {"format": ".png", "bit_depth": 8, "icc_profile": None}

        result = pe.execute(sample_rgb_uint8, config, meta)
        assert result["image"].shape == sample_rgb_uint8.shape

    def test_history_round_trip(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager

        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        hm.add_version(sample_rgb_uint8, {"test": True}, bit_depth=8)

        loaded = hm.get_version_image(1)
        assert loaded is not None
        np.testing.assert_array_equal(loaded, sample_rgb_uint8)

    def test_preset_to_config(self):
        from upscaler.presets.loader import PresetLoader
        loader = PresetLoader()
        preset = loader.load("Fast Preview 2x")
        assert preset is not None
        assert preset["scale"] == 2
        assert preset["pipeline"]["scale"]["plugin"] == "Lanczos"

    def test_registry_discovers_all_builtin_plugins(self):
        from upscaler.plugins.registry import PluginRegistry
        reg = PluginRegistry()
        reg.discover_builtin()
        upscalers = reg.list_plugins("upscaler")
        denoisers = reg.list_plugins("denoiser")
        adjusters = reg.list_plugins("adjuster")
        assert len(upscalers) >= 11  # 5 AI + 6 traditional
        assert len(denoisers) >= 6   # 2 AI + 4 traditional
        assert len(adjusters) >= 7
