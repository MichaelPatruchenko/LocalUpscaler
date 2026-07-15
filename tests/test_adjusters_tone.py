"""Тональные корректоры: Auto Levels, Shadows/Highlights, Clarity + common."""
import cv2
import numpy as np
import pytest


@pytest.fixture
def scene():
    rng = np.random.default_rng(42)
    img = rng.integers(70, 180, (96, 96, 3), dtype=np.uint8)  # узкая гистограмма
    img[:32, :32] = 25    # тёмная зона
    img[-32:, -32:] = 235  # светлая зона
    return img


def _run(plugin_cls, img, params):
    p = plugin_cls()
    p.initialize("cpu")
    return p.process(img, params)


class TestCommon:
    def test_split_merge_roundtrip_uint8(self, scene):
        from upscaler.plugins.adjusters.common import split_alpha, merge_alpha
        rgb, alpha, was_u8 = split_alpha(scene)
        assert rgb.dtype == np.float32 and was_u8 and alpha is None
        assert 0.0 <= rgb.min() and rgb.max() <= 1.0
        back = merge_alpha(rgb, alpha, was_u8)
        assert back.dtype == np.uint8
        assert np.abs(back.astype(int) - scene.astype(int)).max() <= 1

    def test_split_merge_preserves_alpha(self, scene):
        from upscaler.plugins.adjusters.common import split_alpha, merge_alpha
        rgba = np.dstack([scene, np.full(scene.shape[:2], 200, np.uint8)])
        rgb, alpha, was_u8 = split_alpha(rgba)
        out = merge_alpha(rgb, alpha, was_u8)
        assert out.shape[2] == 4
        assert np.array_equal(out[:, :, 3], rgba[:, :, 3])

    def test_float_input_stays_float(self, scene):
        from upscaler.plugins.adjusters.common import split_alpha, merge_alpha
        f = scene.astype(np.float32) / 255.0
        rgb, alpha, was_u8 = split_alpha(f)
        assert not was_u8
        assert merge_alpha(rgb, alpha, was_u8).dtype == np.float32


class TestAutoLevels:
    def _cls(self):
        from upscaler.plugins.adjusters.auto_levels import AutoLevelsPlugin
        return AutoLevelsPlugin

    def test_stretches_narrow_histogram(self, scene):
        out = _run(self._cls(), scene, {"strength": 1.0, "clip_percent": 0.5})
        assert out.min() < scene.min() or out.max() > scene.max()
        assert out.shape == scene.shape and out.dtype == np.uint8

    def test_zero_strength_noop(self, scene):
        out = _run(self._cls(), scene, {"strength": 0.0})
        assert np.abs(out.astype(int) - scene.astype(int)).max() <= 1

    def test_float_roundtrip(self, scene):
        f = scene.astype(np.float32) / 255.0
        out = _run(self._cls(), f, {"strength": 0.7})
        assert out.dtype == np.float32
        assert 0.0 <= out.min() and out.max() <= 1.0


class TestShadowsHighlights:
    def _cls(self):
        from upscaler.plugins.adjusters.shadows_highlights import (
            ShadowsHighlightsPlugin)
        return ShadowsHighlightsPlugin

    def test_lifts_shadows(self, scene):
        out = _run(self._cls(), scene, {"shadows": 1.0, "highlights": 0.0})
        dark = scene[:32, :32].mean()
        assert out[:32, :32].mean() > dark + 3

    def test_tames_highlights(self, scene):
        out = _run(self._cls(), scene, {"shadows": 0.0, "highlights": 1.0})
        bright = scene[-32:, -32:].mean()
        assert out[-32:, -32:].mean() < bright - 3

    def test_zero_noop(self, scene):
        out = _run(self._cls(), scene, {"shadows": 0.0, "highlights": 0.0})
        assert np.abs(out.astype(int) - scene.astype(int)).max() <= 1


class TestClarity:
    def _cls(self):
        from upscaler.plugins.adjusters.clarity import ClarityPlugin
        return ClarityPlugin

    def test_increases_local_contrast(self, scene):
        out = _run(self._cls(), scene, {"strength": 1.0, "radius": 30})
        lap_in = cv2.Laplacian(
            cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        lap_out = cv2.Laplacian(
            cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        assert lap_out > lap_in

    def test_zero_noop(self, scene):
        out = _run(self._cls(), scene, {"strength": 0.0})
        assert np.abs(out.astype(int) - scene.astype(int)).max() <= 1

    def test_output_bounded(self, scene):
        out = _run(self._cls(), scene, {"strength": 1.0, "radius": 60})
        assert out.dtype == np.uint8  # кламп внутри merge_alpha
