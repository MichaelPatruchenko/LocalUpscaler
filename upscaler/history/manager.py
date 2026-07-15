"""Disk-backed history manager with session persistence."""

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from upscaler.utils.image_io import write_image, read_image, make_thumbnail


class HistoryManager:
    """Manages versioned history of processed images on disk."""

    def __init__(self, base_dir: Path | None = None, max_entries: int = 50):
        from upscaler.config import HISTORY_DIR
        self.base_dir = base_dir or HISTORY_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.session_id: str | None = None
        self._session_dir: Path | None = None

    def create_session(self) -> str:
        """Create a new history session. Returns session ID."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self._session_dir = self.base_dir / self.session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_id

    def resume_session(self, session_id: str) -> bool:
        """Resume an existing session. Returns True if found."""
        session_dir = self.base_dir / session_id
        if session_dir.exists():
            self.session_id = session_id
            self._session_dir = session_dir
            return True
        return False

    def delete_session(self, session_id: str):
        """Delete a session and all its data from disk."""
        session_dir = self.base_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

    def add_version(self, image: np.ndarray, metadata: dict, bit_depth: int = 8) -> int:
        """Add a new version to history. Returns version number."""
        if not self._session_dir:
            raise RuntimeError("No active session")

        version = self.get_version_count() + 1
        version_dir = self._session_dir / f"v{version:03d}"
        version_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save image in appropriate format
            if bit_depth <= 8:
                img_path = version_dir / "image.png"
                write_image(image, img_path)
            elif bit_depth <= 16:
                img_path = version_dir / "image.tiff"
                write_image(image, img_path, format="tiff")
            else:
                img_path = version_dir / "image.exr"
                write_image(image, img_path, format="exr")

            # Save thumbnail
            thumb = make_thumbnail(image if image.dtype == np.uint8 else
                                   np.clip(image.astype(np.float32) * 255 if image.max() <= 1 else image / 257, 0, 255).astype(np.uint8),
                                   max_size=150)
            thumb_path = version_dir / "thumbnail.png"
            write_image(thumb, thumb_path)

            # Save metadata
            meta = {
                **metadata,
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "width": image.shape[1],
                "height": image.shape[0],
                "bit_depth": bit_depth,
                "format": img_path.suffix,
            }
            (version_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        except OSError as e:
            # Disk full: prune oldest entries and retry once
            logging.warning(f"Disk write failed: {e}. Pruning history and retrying...")
            shutil.rmtree(version_dir, ignore_errors=True)
            self._prune_oldest(count=5)
            version_dir.mkdir(parents=True, exist_ok=True)
            # Retry write — if this fails too, let the error propagate
            if bit_depth <= 8:
                img_path = version_dir / "image.png"
                write_image(image, img_path)
            elif bit_depth <= 16:
                img_path = version_dir / "image.tiff"
                write_image(image, img_path, format="tiff")
            else:
                img_path = version_dir / "image.exr"
                write_image(image, img_path, format="exr")
            thumb = make_thumbnail(image if image.dtype == np.uint8 else
                                   np.clip(image.astype(np.float32) * 255 if image.max() <= 1 else image / 257, 0, 255).astype(np.uint8),
                                   max_size=150)
            write_image(thumb, version_dir / "thumbnail.png")

        # Prune if over limit
        self._prune()
        return version

    def get_version_image(self, version: int) -> np.ndarray | None:
        """Load the full-resolution image for a version."""
        version_dir = self._session_dir / f"v{version:03d}"
        if not version_dir.exists():
            return None
        for ext in ("png", "tiff", "exr"):
            img_path = version_dir / f"image.{ext}"
            if img_path.exists():
                img, _ = read_image(img_path)
                return img
        return None

    def get_thumbnail(self, version: int) -> np.ndarray | None:
        """Load thumbnail for a version."""
        thumb_path = self._session_dir / f"v{version:03d}" / "thumbnail.png"
        if thumb_path.exists():
            img, _ = read_image(thumb_path)
            return img
        return None

    def get_metadata(self, version: int) -> dict | None:
        """Get metadata for a version."""
        meta_path = self._session_dir / f"v{version:03d}" / "metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return None

    def get_version_count(self) -> int:
        """Count versions in current session."""
        if not self._session_dir:
            return 0
        return len([d for d in self._session_dir.iterdir() if d.is_dir() and d.name.startswith("v")])

    def list_versions(self) -> list[int]:
        """Фактические номера версий текущей сессии (сортированные)."""
        if not self._session_dir or not self._session_dir.exists():
            return []
        versions = []
        for d in self._session_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    continue
        return sorted(versions)

    def delete_version(self, version: int):
        """Delete a specific version."""
        version_dir = self._session_dir / f"v{version:03d}"
        if version_dir.exists():
            shutil.rmtree(version_dir)

    def list_sessions(self) -> list[dict]:
        """List all sessions with basic info."""
        sessions = []
        for d in sorted(self.base_dir.iterdir(), reverse=True):
            if d.is_dir() and not d.name.startswith("."):
                versions = len([v for v in d.iterdir() if v.is_dir() and v.name.startswith("v")])
                sessions.append({"id": d.name, "versions": versions, "path": str(d)})
        return sessions

    def cleanup_old_sessions(self, retention_days: int = 7):
        """Delete sessions older than retention_days."""
        cutoff = datetime.now() - timedelta(days=retention_days)
        for d in self.base_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                try:
                    # Parse date from session ID (YYYYMMDD_HHMMSS_xxx)
                    date_str = d.name[:15]
                    session_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    if session_date < cutoff:
                        shutil.rmtree(d)
                except (ValueError, OSError):
                    continue

    def disk_usage_bytes(self) -> int:
        """Total disk usage of all history."""
        total = 0
        for f in self.base_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def _prune(self):
        """Remove oldest versions if over max_entries."""
        if not self._session_dir:
            return
        versions = sorted([d for d in self._session_dir.iterdir()
                          if d.is_dir() and d.name.startswith("v")])
        while len(versions) > self.max_entries:
            shutil.rmtree(versions.pop(0))

    def _prune_oldest(self, count: int = 5):
        """Force-remove the N oldest versions to free disk space."""
        if not self._session_dir:
            return
        versions = sorted([d for d in self._session_dir.iterdir()
                          if d.is_dir() and d.name.startswith("v")])
        for v in versions[:count]:
            shutil.rmtree(v, ignore_errors=True)
