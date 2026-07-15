import numpy as np
import pytest
from PIL import Image

import upscaler.plugins.icedit.icedit as icedit_mod
from upscaler.plugins.icedit.icedit import ICEditPlugin


class _FakePipe:
    """Stands in for a diffusers FluxFillPipeline; fills the right half white."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, image, mask_image, height, width,
                 num_inference_steps, guidance_scale, generator=None,
                 callback_on_step_end=None, **kw):
        self.calls.append({"prompt": prompt, "width": width, "height": height,
                           "steps": num_inference_steps})
        canvas = np.array(image).copy()
        w = canvas.shape[1] // 2
        canvas[:, w:] = 255  # "edited" right half
        return type("R", (), {"images": [Image.fromarray(canvas)]})()


def _img(h=40, w=60):
    rng = np.random.default_rng(1)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_plugin_metadata():
    assert ICEditPlugin.name == "ICEdit"
    assert ICEditPlugin.category == "icedit"


def test_empty_instruction_returns_input_unchanged():
    p = ICEditPlugin()
    p.initialize("cpu")
    img = _img()
    out = p.process(img, {"instruction": "", "variant": "moe"})
    assert np.array_equal(out, img)


def test_process_runs_edit_with_mocked_pipeline(monkeypatch):
    fake = _FakePipe()
    monkeypatch.setattr(icedit_mod, "load_flux_fill",
                        lambda *a, **k: fake)
    p = ICEditPlugin()
    p.initialize("cpu")
    img = _img(40, 60)
    out = p.process(img, {"instruction": "make it brighter", "variant": "moe",
                          "steps": 12})
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # the edited (white) right half resized back -> brighter than the source
    assert out.mean() > img.mean()
    # instruction reached the pipeline via the diptych template
    assert "make it brighter" in fake.calls[0]["prompt"]
    assert fake.calls[0]["width"] == 1024  # 512-width diptych => 2*512


def test_missing_pipeline_skips_gracefully(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no diffusers")
    monkeypatch.setattr(icedit_mod, "load_flux_fill", boom)
    p = ICEditPlugin()
    p.initialize("cpu")
    img = _img()
    out = p.process(img, {"instruction": "x", "variant": "moe"})
    assert np.array_equal(out, img)


def test_float_input_preserved(monkeypatch):
    fake = _FakePipe()
    monkeypatch.setattr(icedit_mod, "load_flux_fill", lambda *a, **k: fake)
    p = ICEditPlugin()
    p.initialize("cpu")
    img = (_img().astype(np.float32) / 255.0)
    out = p.process(img, {"instruction": "edit", "variant": "moe"})
    assert out.dtype == np.float32
    assert out.max() <= 1.0 + 1e-3


def test_icedit_plugin_is_discovered():
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover_builtin()
    names = [p.name for p in reg.list_plugins("icedit")]
    assert "ICEdit" in names
    assert reg.get("ICEdit") is not None


# --- Этап 5: официальные дефолты ---------------------------------------------

def test_schema_defaults_normal_lora_and_guidance_50():
    from upscaler.plugins.icedit.icedit import ICEditPlugin
    schema = ICEditPlugin.params_schema
    assert schema["variant"]["default"] == "normal"
    assert schema["guidance"]["default"] == 50.0
    assert "рекомендуется" in schema["variant"]["labels"]["normal"]
    assert "экспериментально" in schema["variant"]["labels"]["moe"]


def test_process_inline_defaults_are_normal_and_50(monkeypatch):
    import numpy as np
    from upscaler.plugins.icedit.icedit import ICEditPlugin
    plugin = ICEditPlugin()
    seen = {}

    def fake_get_pipe(variant, quant, offload):
        seen["variant"] = variant
        raise RuntimeError("stop here")

    monkeypatch.setattr(plugin, "_get_pipe", fake_get_pipe)
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    plugin.process(img, {"instruction": "test"})
    assert seen["variant"] == "normal"


def test_unavailable_pipeline_logs_error_and_returns_input(monkeypatch, caplog):
    import logging
    import numpy as np
    from upscaler.plugins.icedit.icedit import ICEditPlugin
    plugin = ICEditPlugin()
    monkeypatch.setattr(plugin, "_get_pipe",
                        lambda *a: (_ for _ in ()).throw(
                            RuntimeError("LoRA не применилась")))
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    with caplog.at_level(logging.ERROR, logger="upscaler.plugins.icedit.icedit"):
        out = plugin.process(img, {"instruction": "make it red"})
    assert out is img
    assert any("ICEdit" in r.message for r in caplog.records
               if r.levelno >= logging.ERROR)
