"""Disk-backed store for pipeline step snapshots (variants)."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np

from upscaler.utils.image_io import write_image, read_image, make_thumbnail


class VariantStore:
    """Manages disk storage of pipeline step snapshots with pruning."""

    def __init__(self, base_dir: Path | None = None, max_entries: int = 30):
        if base_dir is None:
            from upscaler.config import APP_DATA_DIR
            base_dir = APP_DATA_DIR / "variants"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.session_id: str | None = None
        self._session_dir: Path | None = None

    def create_session(self) -> str:
        """Create a new variant session. Returns session ID."""
        import uuid
        from datetime import datetime
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self._session_dir = self.base_dir / self.session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_id

    def add(self, image: np.ndarray, metadata: dict) -> int:
        """Add a new variant snapshot. Returns variant ID."""
        if not self._session_dir:
            raise RuntimeError("No active session")

        # Get next ID
        variant_id = self._get_next_id()
        variant_dir = self._session_dir / f"v{variant_id:03d}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save image
            img_path = variant_dir / "image.png"
            write_image(image, img_path)

            # Save thumbnail
            thumb = make_thumbnail(
                image if image.dtype == np.uint8
                else np.clip(image.astype(np.float32) * 255 if image.max() <= 1 else image / 257, 0, 255).astype(np.uint8),
                max_size=120
            )
            thumb_path = variant_dir / "thumb.png"
            write_image(thumb, thumb_path)

            # Save metadata
            meta = {
                **metadata,
                "variant_id": variant_id,
            }
            (variant_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        except OSError as e:
            logging.warning(f"Disk write failed: {e}. Pruning variants and retrying...")
            shutil.rmtree(variant_dir, ignore_errors=True)
            self._prune_oldest(count=5)
            variant_dir.mkdir(parents=True, exist_ok=True)
            # Retry write
            img_path = variant_dir / "image.png"
            write_image(image, img_path)
            thumb = make_thumbnail(
                image if image.dtype == np.uint8
                else np.clip(image.astype(np.float32) * 255 if image.max() <= 1 else image / 257, 0, 255).astype(np.uint8),
                max_size=120
            )
            write_image(thumb, variant_dir / "thumb.png")

        # Prune if over limit
        self._prune()
        return variant_id

    def get_image(self, variant_id: int) -> np.ndarray | None:
        """Load the full-resolution image for a variant."""
        if not self._session_dir:
            return None
        variant_dir = self._session_dir / f"v{variant_id:03d}"
        if not variant_dir.exists():
            return None
        img_path = variant_dir / "image.png"
        if img_path.exists():
            img, _ = read_image(img_path)
            return img
        return None

    def get_thumbnail(self, variant_id: int) -> np.ndarray | None:
        """Load thumbnail for a variant."""
        if not self._session_dir:
            return None
        thumb_path = self._session_dir / f"v{variant_id:03d}" / "thumb.png"
        if thumb_path.exists():
            img, _ = read_image(thumb_path)
            return img
        return None

    def list_ids(self) -> list[int]:
        """List all variant IDs in current session (gap-safe, sorted)."""
        if not self._session_dir or not self._session_dir.exists():
            return []
        ids = []
        for d in self._session_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    ids.append(int(d.name[1:]))
                except ValueError:
                    continue
        return sorted(ids)

    def clear(self):
        """Delete all variants in current session."""
        if self._session_dir and self._session_dir.exists():
            shutil.rmtree(self._session_dir)
            self._session_dir.mkdir(parents=True, exist_ok=True)

    def _get_next_id(self) -> int:
        """Get the next available variant ID."""
        current_ids = self.list_ids()
        if not current_ids:
            return 1
        return max(current_ids) + 1

    def _prune(self):
        """Remove oldest variants if over max_entries."""
        if not self._session_dir:
            return
        ids = self.list_ids()
        while len(ids) > self.max_entries:
            oldest_id = ids.pop(0)
            variant_dir = self._session_dir / f"v{oldest_id:03d}"
            shutil.rmtree(variant_dir, ignore_errors=True)

    def _prune_oldest(self, count: int = 5):
        """Force-remove the N oldest variants to free disk space."""
        if not self._session_dir:
            return
        ids = self.list_ids()
        for variant_id in ids[:count]:
            variant_dir = self._session_dir / f"v{variant_id:03d}"
            shutil.rmtree(variant_dir, ignore_errors=True)
