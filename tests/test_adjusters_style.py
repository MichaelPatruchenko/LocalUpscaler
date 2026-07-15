"""Стилистические корректоры: Optics, Dodge & Burn, Split Toning."""
import cv2
import numpy as np
import pytest


def _run(plugin_cls, img, params):
    p = plugin_cls()
    p.initialize("cpu")
    return p.process(img, params)


class TestOptics:
    def _cls(self):
        from upscaler.plugins.adjusters.optics import OpticsPlugin
        return OpticsPlugin

    def test_positive_vignette_brightens_corners(self):
        img = np.full((96, 96, 3), 120, np.uint8)
        out = _run(self._cls(), img, {"vignette": 1.0, "ca": 0.0})
        corner = out[:8, :8].mean()
        center = out[44:52, 44:52].mean()
        assert corner > center + 5

    def test_ca_shifts_channels_at_edges(self):
        img = np.zeros((96, 96, 3), np.uint8)
        img[:, 70:74] = 255  # вертикальная белая полоса не в центре
        out = _run(self._cls(), img, {"vignette": 0.0, "ca": 1.0})
        # После поканального масштабирования R и B смещаются в разные стороны
        assert not np.array_equal(out[..., 0], out[..., 2])

    def test_zero_noop(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        out = _run(self._cls(), img, {"vignette": 0.0, "ca": 0.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


class TestDodgeBurn:
    def _cls(self):
        from upscaler.plugins.adjusters.dodge_burn import DodgeBurnPlugin
        return DodgeBurnPlugin

    def _scene(self):
        img = np.full((96, 96, 3), 128, np.uint8)
        img[:32] = 40      # тени
        img[-32:] = 220    # света
        return img

    def test_shadows_up_highlights_down(self):
        img = self._scene()
        out = _run(self._cls(), img, {"strength": 1.0})
        assert out[:32].mean() > img[:32].mean()
        assert out[-32:].mean() < img[-32:].mean()

    def test_monotonic_in_strength(self):
        img = self._scene()
        w = _run(self._cls(), img, {"strength": 0.3})[:32].mean()
        s = _run(self._cls(), img, {"strength": 1.0})[:32].mean()
        assert s >= w

    def test_zero_noop(self):
        img = self._scene()
        out = _run(self._cls(), img, {"strength": 0.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


class TestSplitTone:
    def _cls(self):
        from upscaler.plugins.adjusters.split_tone import SplitTonePlugin
        return SplitTonePlugin

    def _scene(self):
        img = np.full((96, 96, 3), 128, np.uint8)
        img[:32] = 50
        img[-32:] = 210
        return img

    def test_tints_shadows_and_highlights_differently(self):
        img = self._scene()
        # тени в синий (215), света в оранжевый (45)
        out = _run(self._cls(), img, {"shadow_hue": 215.0,
                                      "highlight_hue": 45.0,
                                      "saturation": 1.0, "balance": 0.0})
        sh = out[:32].reshape(-1, 3).mean(axis=0)
        hi = out[-32:].reshape(-1, 3).mean(axis=0)
        assert sh[2] > sh[0]          # тени синее
        assert hi[0] > hi[2]          # света теплее

    def test_zero_saturation_noop(self):
        img = self._scene()
        out = _run(self._cls(), img, {"saturation": 0.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1

    def test_luminance_roughly_preserved(self):
        img = self._scene()
        out = _run(self._cls(), img, {"shadow_hue": 215.0,
                                      "highlight_hue": 45.0,
                                      "saturation": 1.0})
        g_in = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()
        g_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).mean()
        assert abs(g_in - g_out) < 6
