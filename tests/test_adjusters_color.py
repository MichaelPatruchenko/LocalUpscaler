"""Цветовые корректоры: Dehaze, Vibrance, White Balance."""
import cv2
import numpy as np
import pytest


def _run(plugin_cls, img, params):
    p = plugin_cls()
    p.initialize("cpu")
    return p.process(img, params)


def _scene(seed=42):
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 226, (96, 96, 3), dtype=np.uint8)
    return img


class TestDehaze:
    def _cls(self):
        from upscaler.plugins.adjusters.dehaze import DehazePlugin
        return DehazePlugin

    def test_reduces_dark_channel_on_hazy(self):
        base = _scene()
        hazy = np.clip(base.astype(np.float64) * 0.55 + 255 * 0.4,
                       0, 255).astype(np.uint8)
        out = _run(self._cls(), hazy, {"strength": 1.0})
        assert out.min(axis=2).mean() < hazy.min(axis=2).mean() - 5

    def test_zero_noop(self):
        img = _scene()
        out = _run(self._cls(), img, {"strength": 0.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1

    def test_float_roundtrip(self):
        f = _scene().astype(np.float32) / 255.0
        out = _run(self._cls(), f, {"strength": 0.6})
        assert out.dtype == np.float32 and out.max() <= 1.0


class TestVibrance:
    def _cls(self):
        from upscaler.plugins.adjusters.vibrance import VibrancePlugin
        return VibrancePlugin

    def test_boosts_dull_more_than_saturated(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[:2] = (140, 120, 120)   # тусклый красноватый (низкая насыщ.)
        img[2:] = (230, 40, 40)     # насыщенный красный
        out = _run(self._cls(), img, {"strength": 1.0})

        def sat(x):
            hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
            return hsv[..., 1].astype(float)
        gain_dull = sat(out)[:2].mean() - sat(img)[:2].mean()
        gain_sat = sat(out)[2:].mean() - sat(img)[2:].mean()
        assert gain_dull > gain_sat

    def test_gray_stays_gray(self):
        img = np.full((8, 8, 3), 128, np.uint8)
        out = _run(self._cls(), img, {"strength": 1.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1

    def test_skin_protected(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[:2] = (200, 150, 120)   # кожный тон (оранжевый hue)
        img[2:] = (120, 150, 200)   # та же насыщ., синий hue
        out = _run(self._cls(), img, {"strength": 1.0})

        def sat(x):
            return cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].astype(float)
        gain_skin = sat(out)[:2].mean() - sat(img)[:2].mean()
        gain_blue = sat(out)[2:].mean() - sat(img)[2:].mean()
        assert gain_skin < gain_blue

    def test_negative_strength_desaturates(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[:] = (200, 80, 80)
        out = _run(self._cls(), img, {"strength": -1.0})
        s_in = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1].mean()
        s_out = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)[..., 1].mean()
        assert s_out < s_in


class TestWhiteBalance:
    def _cls(self):
        from upscaler.plugins.adjusters.white_balance import WhiteBalancePlugin
        return WhiteBalancePlugin

    def test_removes_cast_with_neutral_patch(self):
        rng = np.random.default_rng(3)
        img = rng.integers(60, 200, (96, 96, 3), dtype=np.uint8)
        img[:32, :32] = (128, 128, 128)  # серая карта
        cast = img.astype(np.float64) * np.array([1.25, 1.0, 0.85])
        cast = np.clip(cast, 0, 255).astype(np.uint8)
        out = _run(self._cls(), cast, {"strength": 1.0})
        means_in = cast[:32, :32].reshape(-1, 3).mean(axis=0)
        means_out = out[:32, :32].reshape(-1, 3).mean(axis=0)
        assert means_out.std() < means_in.std()  # серая карта стала нейтральнее

    def test_zero_noop(self):
        img = _scene()
        out = _run(self._cls(), img, {"strength": 0.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 1

    def test_neutral_image_barely_changes(self):
        img = np.full((32, 32, 3), 120, np.uint8)
        out = _run(self._cls(), img, {"strength": 1.0})
        assert np.abs(out.astype(int) - img.astype(int)).max() <= 2
