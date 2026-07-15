"""Application configuration: paths, defaults, settings persistence."""

import json
import logging
from pathlib import Path

APP_DATA_DIR = Path.home() / ".upscaler"
MODELS_DIR = APP_DATA_DIR / "models"
HISTORY_DIR = APP_DATA_DIR / "history"
PRESETS_USER_DIR = APP_DATA_DIR / "presets"
SETTINGS_FILE = APP_DATA_DIR / "settings.json"
LOG_FILE = APP_DATA_DIR / "upscaler.log"

SUPPORTED_INPUT_FORMATS = {
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
    ".exr", ".hdr", ".cr2", ".nef", ".arw",
}

SUPPORTED_OUTPUT_FORMATS = {"png", "jpg", "jpeg", "tiff", "exr"}

SCALE_FACTORS = [2, 4, 8, 16]

DEFAULT_SETTINGS = {
    "gpu_device": "auto",
    "prefer_gpu_denoise": True,
    "default_output_format": "png",
    "default_output_dir": "",
    "max_history_entries": 50,
    "history_retention_days": 7,
    "tile_size": 512,
    "tile_overlap": 32,
    "model_cache_dir": str(MODELS_DIR),
    "theme": "system",
    "language": "ru",
    # LLM advisor: refine auto-config parameters with a local vision/text GGUF
    # model. Falls back to pure algorithmic config when unavailable.
    "use_llm_advisor": True,
}


def ensure_dirs():
    """Create application directories if they don't exist."""
    for d in (APP_DATA_DIR, MODELS_DIR, HISTORY_DIR, PRESETS_USER_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging():
    """Configure file logging to ~/.upscaler/upscaler.log."""
    ensure_dirs()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_settings() -> dict:
    """Load settings from disk, creating defaults if missing."""
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, "r") as f:
            saved = json.load(f)
        merged = {**DEFAULT_SETTINGS, **saved}
        return merged
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_settings(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict):
    """Persist settings to disk."""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
