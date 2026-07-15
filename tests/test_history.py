import numpy as np
import pytest
import json
import time
from pathlib import Path


class TestHistoryManager:
    def test_create_session(self, tmp_path):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        session_id = hm.create_session()
        assert session_id is not None
        assert (tmp_path / session_id).exists()

    def test_add_version(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        version = hm.add_version(sample_rgb_uint8, {"pipeline": "test"}, bit_depth=8)
        assert version == 1
        assert hm.get_version_count() == 1

    def test_get_version_image(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        hm.add_version(sample_rgb_uint8, {"pipeline": "test"}, bit_depth=8)
        img = hm.get_version_image(1)
        assert img.shape == sample_rgb_uint8.shape

    def test_get_thumbnail(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
        thumb = hm.get_thumbnail(1)
        assert thumb is not None
        assert max(thumb.shape[:2]) <= 150

    def test_revert_to_version(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
        bright = np.clip(sample_rgb_uint8.astype(int) + 50, 0, 255).astype(np.uint8)
        hm.add_version(bright, {}, bit_depth=8)
        img = hm.get_version_image(1)
        np.testing.assert_array_equal(img, sample_rgb_uint8)

    def test_delete_version(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
        hm.delete_version(1)
        assert hm.get_version_count() == 0

    def test_max_entries_prunes_oldest(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path, max_entries=3)
        hm.create_session()
        for i in range(5):
            hm.add_version(sample_rgb_uint8, {"step": i}, bit_depth=8)
        assert hm.get_version_count() == 3

    def test_list_sessions(self, tmp_path):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        sessions = hm.list_sessions()
        assert len(sessions) >= 1

    def test_disk_usage(self, tmp_path, sample_rgb_uint8):
        from upscaler.history.manager import HistoryManager
        hm = HistoryManager(base_dir=tmp_path)
        hm.create_session()
        hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
        usage = hm.disk_usage_bytes()
        assert usage > 0


def test_list_versions_reflects_gaps(tmp_path, sample_rgb_uint8):
    from upscaler.history.manager import HistoryManager
    hm = HistoryManager(base_dir=tmp_path)

    # Нет сессии/версий — пустой список
    assert hm.list_versions() == []

    hm.create_session()
    assert hm.list_versions() == []

    hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
    hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
    hm.add_version(sample_rgb_uint8, {}, bit_depth=8)
    assert hm.list_versions() == [1, 2, 3]

    hm.delete_version(2)
    assert hm.list_versions() == [1, 3]
