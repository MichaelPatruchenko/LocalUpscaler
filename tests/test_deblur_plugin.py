import cv2
import numpy as np
import pytest
import upscaler.plugins.deblur.smartdeblur as sd
from upscaler.plugins.deblur.smartdeblur import SmartDeblurPlugin, _deblur_quality_ok


def _blurred_uint8(size=128):
    rng = np.random.default_rng(3)
    img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    img[size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 230
    return cv2.GaussianBlur(img, (0, 0), 2.0)


def _sharpness(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    return cv2.Laplacian(g.astype(np.float64), cv2.CV_64F).var()


def test_plugin_metadata():
    assert SmartDeblurPlugin.name == "SmartDeblur"
    assert SmartDeblurPlugin.category == "deblur"


def test_deblur_plugin_is_discovered():
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover_builtin()
    names = [p.name for p in reg.list_plugins("deblur")]
    assert "SmartDeblur" in names
    assert reg.get("SmartDeblur") is not None


def test_manual_deblur_sharpens_uint8():
    p = SmartDeblurPlugin()
    p.initialize("cpu")
    blurred = _blurred_uint8()
    params = {"auto": False, "blur_type": "gaussian", "radius": 2.0,
              "smooth": 30, "method": "wiener"}
    out = p.process(blurred, params)
    assert out.dtype == np.uint8
    assert out.shape == blurred.shape
    assert _sharpness(out) > _sharpness(blurred)


def test_auto_deblur_runs():
    p = SmartDeblurPlugin()
    p.initialize("cpu")
    blurred = _blurred_uint8()
    out = p.process(blurred, {"auto": True, "method": "wiener"})
    assert out.shape == blurred.shape
    # auto mode quality is data-dependent; just verify it runs cleanly
    assert np.isfinite(out).all()


def test_float_input_preserved():
    p = SmartDeblurPlugin()
    p.initialize("cpu")
    blurred = _blurred_uint8().astype(np.float32) / 255.0
    out = p.process(blurred, {"auto": False, "blur_type": "gaussian",
                              "radius": 2.0, "smooth": 30, "method": "wiener"})
    assert out.dtype == np.float32
    assert out.max() <= 1.0 + 1e-3


def test_rgba_alpha_preserved():
    p = SmartDeblurPlugin()
    p.initialize("cpu")
    rgb = _blurred_uint8()
    alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 128, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    out = p.process(rgba, {"auto": False, "blur_type": "gaussian",
                           "radius": 2.0, "smooth": 30, "method": "wiener"})
    assert out.shape == rgba.shape
    # alpha channel must pass through untouched
    assert np.array_equal(out[:, :, 3], rgba[:, :, 3])


def test_tv_cancel_callback_honored():
    p = SmartDeblurPlugin()
    p.initialize("cpu")
    blurred = _blurred_uint8()
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 5

    out = p.process(blurred, {
        "auto": False, "blur_type": "gaussian", "radius": 2.0, "smooth": 30,
        "method": "tv", "tv_iterations": 500, "_cancel_cb": cancel,
    })
    assert out.shape == blurred.shape
    assert calls["n"] > 0  # callback was actually consulted


def test_smartdeblur_passes_device_to_deconvolve(monkeypatch):
    import numpy as np
    import upscaler.plugins.deblur.smartdeblur as sd

    seen = {}

    def _fake_deconvolve(channel, kernel, **kwargs):
        seen["device"] = kwargs.get("device")
        return channel  # identity; we only care about the device argument

    monkeypatch.setattr(sd, "deconvolve", _fake_deconvolve)

    plugin = sd.SmartDeblurPlugin()
    plugin.initialize("cuda:0")
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    plugin.process(img, {"auto": False, "method": "wiener"})
    assert seen["device"] == "cuda:0"


def test_quality_ok_true_when_sharper_no_overshoot():
    rng = np.random.default_rng(0)
    before = np.full((64, 64), 0.5, dtype=np.float64)
    after = np.clip(before + rng.normal(0, 0.1, before.shape), 0, 1)  # more detail
    assert _deblur_quality_ok(before, after, after) is True


def test_quality_ok_false_when_not_sharper():
    rng = np.random.default_rng(1)
    before = np.clip(rng.random((64, 64)), 0, 1)         # detailed
    after = np.full((64, 64), before.mean(), dtype=np.float64)  # flat = blurrier
    assert _deblur_quality_ok(before, after, after) is False


def test_quality_ok_false_on_heavy_overshoot():
    rng = np.random.default_rng(2)
    before = np.clip(rng.random((64, 64)), 0, 1)
    sharper = np.clip(before * 1.5, 0, 1)
    pre_clamp = before * 3.0 - 1.0  # massive ringing/overshoot outside [0,1]
    assert _deblur_quality_ok(before, sharper, pre_clamp) is False


def test_process_reverts_to_original_when_deconvolve_blurs(monkeypatch):
    # Force deconvolve to return a blurrier channel -> safeguard reverts.
    monkeypatch.setattr(sd, "deconvolve",
                        lambda ch, *a, **k: np.full_like(ch, float(ch.mean())))
    rng = np.random.default_rng(5)
    img = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    plugin = SmartDeblurPlugin()
    plugin.initialize("cpu")
    out = plugin.process(img, {"auto": False, "method": "wiener", "radius": 3.0})
    assert np.array_equal(out, img)  # original returned unchanged


def test_process_keeps_result_when_deconvolve_sharpens(monkeypatch):
    # Force deconvolve to return a sharper channel -> safeguard keeps it.
    def _sharpen(ch, *a, **k):
        import cv2
        blur = cv2.GaussianBlur(ch, (0, 0), 1.0)
        return np.clip(ch + 0.6 * (ch - blur), 0, 1)
    monkeypatch.setattr(sd, "deconvolve", _sharpen)
    rng = np.random.default_rng(6)
    img = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
    plugin = SmartDeblurPlugin()
    plugin.initialize("cpu")
    out = plugin.process(img, {"auto": False, "method": "wiener", "radius": 3.0})
    assert not np.array_equal(out, img)  # a (sharper) result was kept
