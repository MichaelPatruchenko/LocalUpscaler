"""VariantStore: диск-хранилище промежуточных вариантов обработки."""
import numpy as np


def _store(tmp_path):
    from upscaler.history.variant_store import VariantStore
    s = VariantStore(base_dir=tmp_path)
    s.create_session()
    return s


def _img(v):
    return np.full((32, 32, 3), v, np.uint8)


def test_add_and_get(tmp_path):
    s = _store(tmp_path)
    i = s.add(_img(100), {"step": "denoise"})
    assert s.get_image(i) is not None
    assert s.get_image(i).shape == (32, 32, 3)
    assert s.get_thumbnail(i) is not None


def test_list_ids_gap_safe(tmp_path):
    s = _store(tmp_path)
    ids = [s.add(_img(v), {}) for v in (10, 20, 30)]
    assert s.list_ids() == sorted(ids)


def test_clear(tmp_path):
    s = _store(tmp_path)
    s.add(_img(1), {})
    s.clear()
    assert s.list_ids() == []


def test_prune_by_max_entries(tmp_path):
    from upscaler.history.variant_store import VariantStore
    s = VariantStore(base_dir=tmp_path, max_entries=3)
    s.create_session()
    for v in range(6):
        s.add(_img(v), {})
    assert len(s.list_ids()) <= 3


def test_no_session_returns_empty(tmp_path):
    from upscaler.history.variant_store import VariantStore
    s = VariantStore(base_dir=tmp_path)
    assert s.list_ids() == []
