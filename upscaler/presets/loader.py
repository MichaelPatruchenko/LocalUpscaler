"""Preset loading, saving, and management."""

import json
from pathlib import Path

BUILTIN_DIR = Path(__file__).parent / "builtin"


class PresetLoader:
    """Load and manage pipeline presets."""

    def __init__(self, user_dir: Path | None = None):
        from upscaler.config import PRESETS_USER_DIR
        self.user_dir = user_dir or PRESETS_USER_DIR
        self.user_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[dict]:
        """List all available presets (builtin + user)."""
        presets = []
        # Builtin
        if BUILTIN_DIR.exists():
            for f in sorted(BUILTIN_DIR.glob("*.json")):
                try:
                    presets.append(json.loads(f.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    continue
        # User
        for f in sorted(self.user_dir.glob("*.json")):
            try:
                presets.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue
        return presets

    def load(self, name: str) -> dict | None:
        """Load a preset by name."""
        for preset in self.list_presets():
            if preset.get("name") == name:
                return preset
        return None

    def save(self, preset: dict):
        """Save a user preset."""
        name = preset["name"]
        filename = name.lower().replace(" ", "_").replace("/", "_") + ".json"
        path = self.user_dir / filename
        path.write_text(json.dumps(preset, indent=2), encoding="utf-8")

    def delete(self, name: str) -> bool:
        """Delete a user preset by name. Returns True if found and deleted."""
        for f in self.user_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("name") == name:
                    f.unlink()
                    return True
            except (json.JSONDecodeError, OSError):
                continue
        return False
