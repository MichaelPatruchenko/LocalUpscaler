import pytest
import json
from pathlib import Path


class TestPresetLoader:
    def test_list_builtin_presets(self):
        from upscaler.presets.loader import PresetLoader
        loader = PresetLoader()
        presets = loader.list_presets()
        assert len(presets) >= 7
        names = [p["name"] for p in presets]
        assert "Photo Realistic 4x" in names
        assert "Enhance Only" in names

    def test_load_preset_by_name(self):
        from upscaler.presets.loader import PresetLoader
        loader = PresetLoader()
        preset = loader.load("Photo Realistic 4x")
        assert preset is not None
        assert "pipeline" in preset
        assert "scale" in preset

    def test_save_custom_preset(self, tmp_path):
        from upscaler.presets.loader import PresetLoader
        loader = PresetLoader(user_dir=tmp_path)
        custom = {"name": "My Custom", "scale": 2, "pipeline": {"scale": {"plugin": "Lanczos"}}}
        loader.save(custom)
        loaded = loader.load("My Custom")
        assert loaded["scale"] == 2

    def test_delete_custom_preset(self, tmp_path):
        from upscaler.presets.loader import PresetLoader
        loader = PresetLoader(user_dir=tmp_path)
        custom = {"name": "Temp", "scale": 2, "pipeline": {}}
        loader.save(custom)
        assert loader.load("Temp") is not None
        loader.delete("Temp")
        assert loader.load("Temp") is None

    def test_builtin_presets_are_valid_json(self):
        from upscaler.presets.loader import PresetLoader
        loader = PresetLoader()
        for preset in loader.list_presets():
            assert "name" in preset
            assert "pipeline" in preset
