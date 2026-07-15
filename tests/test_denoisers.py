"""Tests for traditional denoiser plugins."""
import numpy as np
import pytest


@pytest.fixture
def noisy_image():
    rng = np.random.default_rng(42)
    clean = np.full((64, 64, 3), 128, dtype=np.uint8)
    noise = rng.integers(-30, 30, clean.shape, dtype=np.int16)
    return np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)


class TestNLMeans:
    def test_output_shape(self, noisy_image):
        from upscaler.plugins.denoisers.nl_means import NLMeansPlugin
        p = NLMeansPlugin()
        p.initialize("cpu")
        result = p.process(noisy_image, {"strength": 10})
        assert result.shape == noisy_image.shape

    def test_reduces_noise(self, noisy_image):
        from upscaler.plugins.denoisers.nl_means import NLMeansPlugin
        p = NLMeansPlugin()
        p.initialize("cpu")
        result = p.process(noisy_image, {"strength": 10})
        assert np.std(result.astype(float)) < np.std(noisy_image.astype(float))


class TestBilateral:
    def test_output_shape(self, noisy_image):
        from upscaler.plugins.denoisers.bilateral import BilateralPlugin
        p = BilateralPlugin()
        p.initialize("cpu")
        result = p.process(noisy_image, {})
        assert result.shape == noisy_image.shape


class TestWavelet:
    def test_output_shape(self, noisy_image):
        pywt = pytest.importorskip("pywt")  # noqa: F841
        from upscaler.plugins.denoisers.wavelet import WaveletPlugin
        p = WaveletPlugin()
        p.initialize("cpu")
        result = p.process(noisy_image, {"strength": 0.5})
        assert result.shape == noisy_image.shape


class TestBM3D:
    def test_output_shape(self, noisy_image):
        pytest.importorskip("bm3d")
        from upscaler.plugins.denoisers.bm3d_plugin import BM3DPlugin
        p = BM3DPlugin()
        p.initialize("cpu")
        result = p.process(noisy_image, {"sigma": 25})
        assert result.shape == noisy_image.shape


class TestSCUNet:
    def test_metadata(self):
        from upscaler.plugins.denoisers.scunet import SCUNetPlugin
        assert SCUNetPlugin.name == "SCUNet"
        assert SCUNetPlugin.category == "denoiser"
        assert SCUNetPlugin.gpu_memory_mb > 0


class TestNAFNet:
    def test_metadata(self):
        from upscaler.plugins.denoisers.nafnet import NAFNetPlugin
        assert NAFNetPlugin.name == "NAFNet"
        assert NAFNetPlugin.category == "denoiser"
        assert NAFNetPlugin.gpu_memory_mb > 0


def test_bm3d_fast_param_selects_lc_profile(monkeypatch):
    import sys
    import types
    import numpy as np

    captured = {}

    fake = types.ModuleType("bm3d")

    class BM3DProfileLC:
        pass

    def fake_bm3d(z, sigma, profile="np", **kwargs):
        captured["profile"] = profile
        return z

    fake.BM3DProfileLC = BM3DProfileLC
    fake.bm3d = fake_bm3d
    monkeypatch.setitem(sys.modules, "bm3d", fake)

    from upscaler.plugins.denoisers.bm3d_plugin import BM3DPlugin
    plugin = BM3DPlugin()
    plugin.initialize("cpu")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    plugin.process(img, {"sigma": 25})  # fast defaults to True
    assert isinstance(captured["profile"], BM3DProfileLC)

    plugin.process(img, {"sigma": 25, "fast": False})
    assert captured["profile"] == "np"
