import pytest
from upscaler.engine.denoise_routing import cuda_available, substitute_gpu_denoise


def test_cuda_available_cpu_is_false():
    assert cuda_available("cpu") is False


def test_cuda_available_respects_torch(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert cuda_available("cuda") is True
    assert cuda_available("auto") is True
    assert cuda_available("cuda:0") is True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cuda_available("auto") is False


def test_substitute_bm3d_to_scunet():
    out = substitute_gpu_denoise({"BM3D": {"sigma": 25}}, True, True)
    assert "BM3D" not in out
    assert out["SCUNet"]["strength"] == 0.5


def test_substitute_disabled_unchanged():
    cfg = {"BM3D": {"sigma": 25}}
    assert substitute_gpu_denoise(cfg, True, False) == cfg


def test_substitute_no_gpu_unchanged():
    cfg = {"BM3D": {"sigma": 25}}
    assert substitute_gpu_denoise(cfg, False, True) == cfg


def test_substitute_non_bm3d_unchanged():
    cfg = {"NL-Means": {"strength": 10}}
    assert substitute_gpu_denoise(cfg, True, True) == cfg


def test_substitute_keeps_existing_scunet():
    out = substitute_gpu_denoise(
        {"BM3D": {"sigma": 50}, "SCUNet": {"strength": 0.3}}, True, True)
    assert "BM3D" not in out
    assert out["SCUNet"]["strength"] == 0.3


def test_sigma_mapping_bounds():
    assert substitute_gpu_denoise({"BM3D": {"sigma": 10}}, True, True)["SCUNet"]["strength"] == 0.2
    assert substitute_gpu_denoise({"BM3D": {"sigma": 75}}, True, True)["SCUNet"]["strength"] == 1.0
    assert substitute_gpu_denoise({"BM3D": {"sigma": 1}}, True, True)["SCUNet"]["strength"] == 0.1


def test_input_not_mutated():
    cfg = {"BM3D": {"sigma": 25}}
    substitute_gpu_denoise(cfg, True, True)
    assert "BM3D" in cfg


def test_pipeline_routes_bm3d_to_scunet_on_gpu(monkeypatch):
    import numpy as np
    import upscaler.engine.denoise_routing as dr
    from upscaler.engine.pipeline import PipelineExecutor

    ran = []

    def _make_spy(label):
        class _Spy:
            def __init__(self):
                pass
            def initialize(self, device):
                pass
            def process(self, img, params):
                ran.append(label)
                return img
            def cleanup(self):
                pass
        return _Spy

    class _FakeRegistry:
        def __init__(self, mapping):
            self._m = mapping
        def get(self, name):
            return self._m.get(name)
        def list_plugins(self, category):
            return []

    registry = _FakeRegistry({"BM3D": _make_spy("BM3D"),
                              "SCUNet": _make_spy("SCUNet")})
    executor = PipelineExecutor(registry)
    img = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    base_cfg = {"scale": 2, "enhance_only": True, "denoise": {"BM3D": {"sigma": 25}},
                "adjust": {}, "upscale": {}, "post": {}, "prefer_gpu_denoise": True}

    # GPU available -> BM3D routed to SCUNet
    monkeypatch.setattr(dr, "cuda_available", lambda device: True)
    ran.clear()
    executor.execute(img, dict(base_cfg), {"bit_depth": 8}, device="cuda")
    assert "SCUNet" in ran and "BM3D" not in ran

    # No GPU -> BM3D runs as chosen
    monkeypatch.setattr(dr, "cuda_available", lambda device: False)
    ran.clear()
    executor.execute(img, dict(base_cfg), {"bit_depth": 8}, device="cpu")
    assert "BM3D" in ran and "SCUNet" not in ran
