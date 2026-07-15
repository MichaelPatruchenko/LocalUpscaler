import builtins
import sys
import types

import pytest

from upscaler.plugins.icedit import pipeline_loader


def test_load_flux_fill_raises_without_diffusers(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("diffusers"):
            raise ImportError("no diffusers")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Pass a truthy model_manager so execution reaches the diffusers import
    # (the None-guard now short-circuits before imports). object() is enough:
    # the import fails before any model_manager method is called.
    with pytest.raises(Exception):
        pipeline_loader.load_flux_fill("moe", "q4", "model", "cpu", object())


def test_load_flux_fill_raises_without_model_manager():
    with pytest.raises(RuntimeError):
        pipeline_loader.load_flux_fill("moe", "q4", "model", "cpu", None)


class _FakeModelManager:
    """Records which registry keys were resolved; reports everything present."""

    def __init__(self, present):
        self._present = set(present)
        self.requested = []

    def is_downloaded(self, key):
        return key in self._present

    def download(self, key):  # pragma: no cover - present fixture never downloads
        self._present.add(key)

    def get_model_path(self, key):
        self.requested.append(key)
        return f"/models/{key}"


def _install_fake_ml(monkeypatch):
    """Inject stub diffusers/transformers modules recording assembly calls."""
    calls = {"pipe_kwargs": None, "lora": None, "placement": [], "transformer": None}

    class _FakeComponent:
        def __init__(self, tag):
            self.tag = tag

    class _FakeParam:
        pass

    class _FakeTransformer:
        def __init__(self, names):
            self._names = names

        def named_parameters(self):
            return [(n, _FakeParam()) for n in self._names]

    class _FluxTransformer2DModel:
        @staticmethod
        def from_single_file(path, **kw):
            calls["transformer"] = path
            return _FakeComponent(("transformer", path))

    class _AutoencoderKL:
        @staticmethod
        def from_single_file(path, **kw):
            return _FakeComponent(("vae", path))

    class _GGUFQuantizationConfig:
        def __init__(self, **kw):
            pass

    class _FlowMatchEulerDiscreteScheduler:
        def __init__(self, **kw):
            self.kw = kw

    class _FluxFillPipeline:
        def __init__(self, **kwargs):
            calls["pipe_kwargs"] = kwargs
            # Include a transformer with some LoRA parameters to satisfy
            # _assert_lora_applied verification.
            self.transformer = _FakeTransformer([
                "blocks.0.attn.to_q.lora_A.default.weight",
                "blocks.0.attn.to_q.lora_B.default.weight",
                "blocks.0.attn.to_q.weight",
            ])

        def load_lora_weights(self, path):
            calls["lora"] = path

        def to(self, dev):
            calls["placement"].append(("to", dev))

        def enable_model_cpu_offload(self):
            calls["placement"].append(("model_offload", None))

        def enable_sequential_cpu_offload(self):
            calls["placement"].append(("sequential_offload", None))

    class _FromPretrained:
        @staticmethod
        def from_pretrained(repo, **kw):
            return _FakeComponent(("from_pretrained", repo, kw.get("gguf_file")))

    diffusers = types.ModuleType("diffusers")
    diffusers.FluxFillPipeline = _FluxFillPipeline
    diffusers.FluxTransformer2DModel = _FluxTransformer2DModel
    diffusers.AutoencoderKL = _AutoencoderKL
    diffusers.FlowMatchEulerDiscreteScheduler = _FlowMatchEulerDiscreteScheduler
    diffusers.GGUFQuantizationConfig = _GGUFQuantizationConfig

    transformers = types.ModuleType("transformers")
    transformers.CLIPTextModel = _FromPretrained
    transformers.CLIPTokenizer = _FromPretrained
    transformers.T5EncoderModel = _FromPretrained
    transformers.T5TokenizerFast = _FromPretrained

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_assembly_wires_all_components_and_cpu_skips_offload(monkeypatch):
    calls = _install_fake_ml(monkeypatch)
    mm = _FakeModelManager(present=[
        "FLUX-Fill-GGUF-Q5", "FLUX-VAE", "ICEdit-MoE-LoRA"])

    pipe = pipeline_loader.load_flux_fill("moe", "q4", "model", "cpu", mm)

    kw = calls["pipe_kwargs"]
    # all 7 FluxFill components were supplied
    assert set(kw) == {"scheduler", "vae", "text_encoder", "tokenizer",
                       "text_encoder_2", "tokenizer_2", "transformer"}
    # MoE LoRA was loaded from the resolved local path
    assert calls["lora"] == "/models/ICEdit-MoE-LoRA"
    # a present Q5 transformer is preferred even though "q4" was requested
    assert calls["transformer"] == "/models/FLUX-Fill-GGUF-Q5"
    # CPU machine: the pipeline is left as assembled (already on CPU); no .to()
    # call (it would hit a meta-tensor error on the GGUF transformer) and no
    # GPU-only offload.
    assert calls["placement"] == []


def test_assembly_normal_variant_loads_normal_lora(monkeypatch):
    calls = _install_fake_ml(monkeypatch)
    mm = _FakeModelManager(present=[
        "FLUX-Fill-GGUF-Q4", "FLUX-VAE", "ICEdit-normal-LoRA"])

    pipeline_loader.load_flux_fill("normal", "q4", "model", "cpu", mm)
    assert calls["lora"] == "/models/ICEdit-normal-LoRA"
    assert calls["transformer"] == "/models/FLUX-Fill-GGUF-Q4"


# --- Этап 5: верификация применения LoRA --------------------------------------

class _FakeParam:
    pass


class _FakeTransformer:
    def __init__(self, names):
        self._names = names

    def named_parameters(self):
        return [(n, _FakeParam()) for n in self._names]


class _FakePipe:
    def __init__(self, names):
        self.transformer = _FakeTransformer(names)


def test_assert_lora_applied_counts_lora_params():
    pipe = _FakePipe([
        "blocks.0.attn.to_q.lora_A.default.weight",
        "blocks.0.attn.to_q.lora_B.default.weight",
        "blocks.0.attn.to_q.weight",
    ])
    assert pipeline_loader._assert_lora_applied(pipe, "ICEdit-normal-LoRA") == 2


def test_assert_lora_applied_raises_when_no_lora_layers():
    pipe = _FakePipe(["blocks.0.attn.to_q.weight",
                      "blocks.0.mlp.fc1.weight"])
    with pytest.raises(RuntimeError) as err:
        pipeline_loader._assert_lora_applied(pipe, "ICEdit-MoE-LoRA")
    assert "ICEdit-MoE-LoRA" in str(err.value)


def test_resolve_lora_key_fallback_is_normal():
    # Незнакомый вариант не должен молча уводить на несовместимую MoE.
    assert pipeline_loader._resolve_lora_key("bogus") == "ICEdit-normal-LoRA"
    assert pipeline_loader._resolve_lora_key("normal") == "ICEdit-normal-LoRA"
    assert pipeline_loader._resolve_lora_key("moe") == "ICEdit-MoE-LoRA"
