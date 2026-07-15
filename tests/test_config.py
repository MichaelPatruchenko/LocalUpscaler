import pytest
from pathlib import Path


def test_default_settings_has_required_keys():
    from upscaler.config import DEFAULT_SETTINGS
    required = [
        "gpu_device", "default_output_format", "default_output_dir",
        "max_history_entries", "history_retention_days", "tile_size",
        "tile_overlap", "model_cache_dir", "theme",
    ]
    for key in required:
        assert key in DEFAULT_SETTINGS, f"Missing default setting: {key}"


def test_default_settings_values():
    from upscaler.config import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["gpu_device"] == "auto"
    assert DEFAULT_SETTINGS["default_output_format"] == "png"
    assert DEFAULT_SETTINGS["max_history_entries"] == 50
    assert DEFAULT_SETTINGS["tile_size"] == 512
    assert DEFAULT_SETTINGS["tile_overlap"] == 32


def test_app_dirs_are_paths():
    from upscaler.config import APP_DATA_DIR, MODELS_DIR, HISTORY_DIR, PRESETS_USER_DIR
    assert isinstance(APP_DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    assert isinstance(HISTORY_DIR, Path)
    assert isinstance(PRESETS_USER_DIR, Path)


def test_settings_load_creates_default_if_missing(tmp_path):
    from upscaler.config import load_settings, SETTINGS_FILE
    import upscaler.config as cfg
    original = cfg.SETTINGS_FILE
    cfg.SETTINGS_FILE = tmp_path / "settings.json"
    try:
        settings = load_settings()
        assert settings["gpu_device"] == "auto"
        assert (tmp_path / "settings.json").exists()
    finally:
        cfg.SETTINGS_FILE = original


def test_settings_save_and_reload(tmp_path):
    from upscaler.config import load_settings, save_settings
    import upscaler.config as cfg
    original = cfg.SETTINGS_FILE
    cfg.SETTINGS_FILE = tmp_path / "settings.json"
    try:
        settings = load_settings()
        settings["theme"] = "dark"
        save_settings(settings)
        reloaded = load_settings()
        assert reloaded["theme"] == "dark"
    finally:
        cfg.SETTINGS_FILE = original


def test_default_settings_has_prefer_gpu_denoise():
    from upscaler.config import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS["prefer_gpu_denoise"] is True
