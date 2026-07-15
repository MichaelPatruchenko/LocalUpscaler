"""Tests for upscaler.models.manager.ModelManager."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

torch = pytest.importorskip("torch")


class TestModelManager:
    def test_model_registry_has_entries(self):
        from upscaler.models.manager import ModelManager
        mm = ModelManager(cache_dir=Path("/tmp/test_models"))
        models = mm.list_models()
        assert len(models) > 0
        assert "Real-ESRGAN-x4" in models or "real_esrgan_x4" in [m.lower().replace("-", "_") for m in models]

    def test_is_downloaded_false_for_missing_model(self, tmp_path):
        from upscaler.models.manager import ModelManager, MODEL_REGISTRY, LOCAL_MODELS_DIR
        mm = ModelManager(cache_dir=tmp_path)
        # Find a model whose file doesn't exist locally
        for name, info in MODEL_REGISTRY.items():
            local = LOCAL_MODELS_DIR / info["filename"]
            cache = tmp_path / info["filename"]
            if not local.exists() and not cache.exists():
                assert mm.is_downloaded(name) is False
                return
        pytest.skip("All models are downloaded locally")

    def test_get_model_path(self, tmp_path):
        from upscaler.models.manager import ModelManager
        mm = ModelManager(cache_dir=tmp_path)
        path = mm.get_model_path("Real-ESRGAN-x4")
        assert isinstance(path, Path)

    def test_get_device_auto_returns_string(self):
        from upscaler.models.manager import ModelManager
        mm = ModelManager(cache_dir=Path("/tmp"))
        device = mm.get_device("auto")
        assert device in ("cuda", "cpu")

    def test_get_device_cpu(self):
        from upscaler.models.manager import ModelManager
        mm = ModelManager(cache_dir=Path("/tmp"))
        assert mm.get_device("cpu") == "cpu"

    def test_icedit_models_registered(self):
        from upscaler.models.manager import MODEL_REGISTRY
        for key in [
            "FLUX-Fill-GGUF-Q4", "FLUX-Fill-GGUF-Q5", "FLUX-T5-GGUF",
            "FLUX-CLIP-L", "FLUX-VAE", "ICEdit-MoE-LoRA", "ICEdit-normal-LoRA",
        ]:
            assert key in MODEL_REGISTRY, key
            assert MODEL_REGISTRY[key]["filename"]
            assert "url" in MODEL_REGISTRY[key]
