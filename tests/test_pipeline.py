"""Tests for the six-stage pipeline executor."""
import numpy as np
import pytest


@pytest.fixture
def registry_with_plugins():
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.plugins.upscalers.lanczos import LanczosPlugin
    from upscaler.plugins.adjusters.brightness import BrightnessPlugin
    reg = PluginRegistry()
    reg.register(LanczosPlugin)
    reg.register(BrightnessPlugin)
    return reg


class TestPipelineExecutor:
    def test_basic_upscale_pipeline(self, sample_rgb_uint8, registry_with_plugins):
        from upscaler.engine.pipeline import PipelineExecutor
        pe = PipelineExecutor(registry_with_plugins)
        config = {
            "scale": 2,
            "upscale": {"plugin": "Lanczos"},
            "enhance_only": False,
        }
        result = pe.execute(sample_rgb_uint8, config, meta={"format": ".png", "bit_depth": 8, "icc_profile": None})
        assert result["image"].shape == (128, 128, 3)
        assert "metrics" in result

    def test_enhance_only_skips_scaling(self, sample_rgb_uint8, registry_with_plugins):
        from upscaler.engine.pipeline import PipelineExecutor
        pe = PipelineExecutor(registry_with_plugins)
        config = {
            "scale": 2,
            "enhance_only": True,
            "adjust": {"Brightness": {"value": 10}},
        }
        result = pe.execute(sample_rgb_uint8, config, meta={"format": ".png", "bit_depth": 8, "icc_profile": None})
        assert result["image"].shape == (64, 64, 3)

    def test_cancel_event_stops_pipeline(self, sample_rgb_uint8, registry_with_plugins):
        import threading
        from upscaler.engine.pipeline import PipelineExecutor, PipelineCancelled
        pe = PipelineExecutor(registry_with_plugins)
        cancel = threading.Event()
        cancel.set()
        config = {"scale": 2, "upscale": {"plugin": "Lanczos"}, "enhance_only": False}
        with pytest.raises(PipelineCancelled):
            pe.execute(sample_rgb_uint8, config, meta={"format": ".png", "bit_depth": 8, "icc_profile": None}, cancel_event=cancel)

    def test_progress_callback(self, sample_rgb_uint8, registry_with_plugins):
        from upscaler.engine.pipeline import PipelineExecutor
        pe = PipelineExecutor(registry_with_plugins)
        stages = []

        def on_progress(stage, pct, msg):
            stages.append(stage)

        config = {"scale": 2, "upscale": {"plugin": "Lanczos"}, "enhance_only": False}
        pe.execute(sample_rgb_uint8, config, meta={"format": ".png", "bit_depth": 8, "icc_profile": None}, progress_cb=on_progress)
        assert "АНАЛИЗ" in stages
        assert "ПРОВЕРКА" in stages


class TestStepOrdering:
    def test_resolve_order_defaults_when_missing(self):
        from upscaler.engine.pipeline import PipelineExecutor
        from upscaler.plugins.registry import PluginRegistry
        pe = PipelineExecutor(PluginRegistry())
        assert pe.resolve_order(None) == pe.DEFAULT_ORDER
        assert pe.resolve_order("nope") == pe.DEFAULT_ORDER

    def test_resolve_order_honors_sequence_and_appends_missing(self):
        from upscaler.engine.pipeline import PipelineExecutor
        from upscaler.plugins.registry import PluginRegistry
        pe = PipelineExecutor(PluginRegistry())
        seq = pe.resolve_order(["sharpen", "upscale", "bogus", "sharpen"])
        # requested-known-first, de-duped, unknown dropped
        assert seq[:2] == ["sharpen", "upscale"]
        # every default step still present exactly once
        assert sorted(seq) == sorted(pe.DEFAULT_ORDER)

    def test_custom_order_changes_step_sequence(self, sample_rgb_uint8):
        """A custom order runs steps in the requested sequence."""
        from upscaler.engine.pipeline import PipelineExecutor
        from upscaler.plugins.registry import PluginRegistry
        from upscaler.plugins.upscalers.lanczos import LanczosPlugin
        from upscaler.plugins.adjusters.brightness import BrightnessPlugin
        reg = PluginRegistry()
        reg.register(LanczosPlugin)
        reg.register(BrightnessPlugin)
        pe = PipelineExecutor(reg)

        seen = []

        def on_progress(stage, pct, msg):
            if msg:
                seen.append(msg)

        config = {
            "scale": 2, "enhance_only": False,
            "upscale": {"plugin": "Lanczos"},
            "adjust": {"Brightness": {"value": 10}},
            "order": ["upscale", "adjust"],  # upscale before adjust (non-default)
        }
        pe.execute(sample_rgb_uint8, config,
                   meta={"format": ".png", "bit_depth": 8, "icc_profile": None},
                   progress_cb=on_progress)
        up_idx = next(i for i, m in enumerate(seen) if "Увеличение" in m)
        adj_idx = next(i for i, m in enumerate(seen) if "Коррекция" in m)
        assert up_idx < adj_idx


def test_per_adjuster_ordering():
    """Individual adjuster tokens position each adjuster independently."""
    import numpy as np
    from upscaler.engine.pipeline import PipelineExecutor
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover_builtin()
    pe = PipelineExecutor(reg)

    seen = []

    def cb(stage, pct, msg):
        if msg.startswith("Коррекция"):
            seen.append(msg)

    img = (np.random.default_rng(1).random((32, 32, 3)) * 255).astype(np.uint8)
    config = {
        "scale": 2, "enhance_only": True,
        "adjust": {
            "Brightness": {"strength": 0.2, "direction": "up"},
            "Saturation": {"strength": 0.2, "direction": "up"},
        },
        "order": ["saturation", "brightness"],  # saturation first (non-default)
    }
    pe.execute(img, config,
               meta={"format": ".png", "bit_depth": 8, "icc_profile": None},
               progress_cb=cb)
    sat_idx = next(i for i, m in enumerate(seen) if "Saturation" in m)
    bri_idx = next(i for i, m in enumerate(seen) if "Brightness" in m)
    assert sat_idx < bri_idx


def test_pipeline_runs_deblur_stage():
    import cv2
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor

    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)

    img = (np.random.default_rng(4).random((64, 64, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    config = {
        "scale": 2, "enhance_only": True,
        "deblur": {"auto": False, "blur_type": "gaussian", "radius": 2.0,
                   "smooth": 30, "method": "wiener"},
        "denoise": {}, "adjust": {}, "upscale": {}, "post": {},
    }
    result = ex.execute(img, config, meta={}, device="cpu")
    out = result["image"]
    assert out is not None and np.isfinite(out).all()
    sharp_in = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float), cv2.CV_64F).var()
    sharp_out = cv2.Laplacian(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(float), cv2.CV_64F).var()
    assert sharp_out > sharp_in


def test_pipeline_runs_icedit_step():
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor
    import upscaler.plugins.icedit.icedit as icedit_mod
    from PIL import Image

    class _FakePipe:
        def __call__(self, prompt, image, mask_image, height, width,
                     num_inference_steps, guidance_scale, generator=None,
                     callback_on_step_end=None, **kw):
            canvas = np.array(image).copy()
            w = canvas.shape[1] // 2
            canvas[:, w:] = 0  # darken right half so the effect is measurable
            return type("R", (), {"images": [Image.fromarray(canvas)]})()

    icedit_mod.load_flux_fill = lambda *a, **k: _FakePipe()

    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)

    img = (np.random.default_rng(4).random((64, 64, 3)) * 255).astype(np.uint8)
    config = {
        "scale": 2, "enhance_only": True,
        "icedit": {"instruction": "darken", "variant": "moe", "steps": 8},
        "denoise": {}, "adjust": {}, "upscale": {}, "post": {},
    }
    result = ex.execute(img, config, meta={}, device="cpu")
    assert result["image"] is not None
    assert result["image"].shape == img.shape
    # the edit darkened the image
    assert result["image"].mean() < img.mean()


def test_pipeline_skips_icedit_without_instruction():
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor

    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)
    img = (np.random.default_rng(5).random((48, 48, 3)) * 255).astype(np.uint8)
    config = {"scale": 2, "enhance_only": True, "icedit": {"instruction": ""},
              "denoise": {}, "adjust": {}, "upscale": {}, "post": {}}
    result = ex.execute(img, config, meta={}, device="cpu")
    assert np.array_equal(result["image"], img)


def test_pipeline_runs_face_restore_step():
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor

    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)

    called = {"n": 0}

    class _FakeFace:
        name = "CodeFormer"
        category = "face"
        supported_scales = []
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params):
            called["n"] += 1
            return image
        def cleanup(self): pass

    reg.register(_FakeFace)

    img = (np.random.default_rng(4).random((48, 48, 3)) * 255).astype(np.uint8)
    config = {"scale": 2, "enhance_only": True,
              "face": {"enabled": True, "fidelity": 0.7},
              "denoise": {}, "adjust": {}, "upscale": {}, "post": {}}
    result = ex.execute(img, config, meta={}, device="cpu")
    assert result["image"] is not None
    assert called["n"] == 1


def test_face_restore_skipped_when_disabled():
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor

    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)

    called = {"n": 0}

    class _FakeFace:
        name = "CodeFormer"
        category = "face"
        supported_scales = []
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params):
            called["n"] += 1
            return image
        def cleanup(self): pass

    reg.register(_FakeFace)

    img = (np.random.default_rng(5).random((48, 48, 3)) * 255).astype(np.uint8)
    config = {"scale": 2, "enhance_only": True,
              "face": {"enabled": False},
              "denoise": {}, "adjust": {}, "upscale": {}, "post": {}}
    result = ex.execute(img, config, meta={}, device="cpu")
    assert result["image"] is not None
    assert called["n"] == 0, "face_restore must NOT run when enabled=False"
    assert np.array_equal(result["image"], img)


def test_face_restore_skipped_when_empty_config():
    """Regression guard: empty face config (checkbox-off) must skip face restore."""
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    from upscaler.engine.pipeline import PipelineExecutor

    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)

    called = {"n": 0}

    class _FakeFace:
        name = "CodeFormer"
        category = "face"
        supported_scales = []
        params_schema = {}
        def initialize(self, device): pass
        def process(self, image, params):
            called["n"] += 1
            return image
        def cleanup(self): pass

    reg.register(_FakeFace)

    img = (np.random.default_rng(6).random((48, 48, 3)) * 255).astype(np.uint8)
    config = {"scale": 2, "enhance_only": True,
              "face": {},
              "denoise": {}, "adjust": {}, "upscale": {}, "post": {}}
    result = ex.execute(img, config, meta={}, device="cpu")
    assert result["image"] is not None
    assert called["n"] == 0, "face_restore must NOT run when face config is empty {}"
    assert np.array_equal(result["image"], img)


# --- Этап 2: подписи шагов и модульный resolve -----------------------------
from upscaler.engine.pipeline import (
    PipelineExecutor, STEP_LABELS, resolve_step_order, step_label,
)


def test_step_labels_cover_default_order():
    for token in PipelineExecutor.DEFAULT_ORDER:
        assert token in STEP_LABELS, f"нет подписи для токена {token}"
        assert step_label(token).strip()


def test_resolve_step_order_none_gives_default():
    assert resolve_step_order(None) == list(PipelineExecutor.DEFAULT_ORDER)


def test_resolve_step_order_prefix_and_completion():
    out = resolve_step_order(["upscale", "denoise", "upscale", "bogus"])
    assert out[0] == "upscale" and out[1] == "denoise"
    assert out.count("upscale") == 1
    assert "bogus" not in out
    assert set(out) == set(PipelineExecutor.DEFAULT_ORDER)


def test_executor_resolve_order_delegates():
    from upscaler.plugins.registry import PluginRegistry
    ex = PipelineExecutor(PluginRegistry())
    assert ex.resolve_order(["colorize"]) == resolve_step_order(["colorize"])


# --- Этап 4: предуменьшение -------------------------------------------------

def _run_pipeline(config, size=64):
    from upscaler.plugins.registry import PluginRegistry
    ex = PipelineExecutor(PluginRegistry())
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    return ex.execute(img, config, {})["image"]


def test_predownscale_first_in_default_order():
    assert PipelineExecutor.DEFAULT_ORDER[0] == "predownscale"
    assert resolve_step_order(None)[0] == "predownscale"


def test_predownscale_resizes_image():
    out = _run_pipeline({"predownscale": {"factor": 0.5}})
    assert out.shape[:2] == (32, 32)


def test_predownscale_disabled_flag_skips():
    out = _run_pipeline({"predownscale": {"factor": 0.5},
                         "predownscale_enabled": False})
    assert out.shape[:2] == (64, 64)


def test_predownscale_invalid_factor_ignored():
    for factor in (0.0, 1.0, 1.5, -2.0):
        out = _run_pipeline({"predownscale": {"factor": factor}})
        assert out.shape[:2] == (64, 64)


def test_no_predownscale_config_no_change():
    out = _run_pipeline({})
    assert out.shape[:2] == (64, 64)


# --- Этап 6A: шаг blend и снимки ---------------------------------------------


def test_blend_last_in_default_order():
    assert PipelineExecutor.DEFAULT_ORDER[-1] == "blend"
    assert "blend" in STEP_LABELS


def test_blend_disabled_is_noop():
    out = _run_pipeline({"blend": {"enabled": False}})
    assert out.shape[:2] == (64, 64)


def test_blend_with_fewer_than_two_candidates_is_noop():
    # Ни один снимковый шаг не выполняется -> только "result" -> no-op
    img_before = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = _run_pipeline({"blend": {"enabled": True}})
    assert np.array_equal(out, img_before)


def test_blend_no_candidates_no_search(monkeypatch):
    """Blend не выдумывает кандидатов: без снимков поиск не вызывается."""
    import upscaler.engine.blend_search as bs
    called = {}
    monkeypatch.setattr(bs, "greedy_blend_search",
                        lambda cands, **kw: called.setdefault(
                            "candidates", sorted(cands)))
    _run_pipeline({"blend": {"enabled": True}})  # пустой реестр: шагов нет
    assert "candidates" not in called


def test_blend_runs_search_with_real_registry(monkeypatch):
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry()
    reg.discover_builtin()
    ex = PipelineExecutor(reg)
    called = {}
    import upscaler.engine.blend_search as bs

    def spy_search(cands, **kw):
        called["candidates"] = sorted(cands)
        return {"base": "result", "layers": [], "score": 1.0}

    monkeypatch.setattr(bs, "greedy_blend_search", spy_search)
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    cfg = {"blend": {"enabled": True}, "post": {"sharpen": 0.8}}
    ex.execute(img, cfg, {})
    # снимок после sharpen + текущий результат = >=2 кандидатов -> поиск был
    assert "candidates" in called
    assert "result" in called["candidates"]
    assert "sharpen" in called["candidates"]


# --- Этап A: 10 новых adjuster-токенов ---------------------------------------

_NEW_ADJUSTER_TOKENS = [
    "optics", "white_balance", "dehaze", "auto_levels",
    "shadows_highlights", "dodge_burn", "vibrance", "split_tone",
    "clarity", "skin_smooth",
]


def test_new_adjuster_tokens_registered():
    for token in _NEW_ADJUSTER_TOKENS:
        assert token in PipelineExecutor.DEFAULT_ORDER, token
        assert token in STEP_LABELS, token
        assert token in PipelineExecutor._ADJUST_TOKEN_PLUGIN, token


def test_new_default_order_positions():
    order = PipelineExecutor.DEFAULT_ORDER
    assert order.index("optics") > order.index("icedit")
    assert order.index("white_balance") < order.index("auto_color")
    assert order.index("skin_smooth") > order.index("face_restore")
    assert order.index("clarity") < order.index("sharpness")
    assert order[-1] == "blend" and order[0] == "predownscale"


# --- Блок G: сбор вариантов ---------------------------------------------------

def test_execute_returns_variants_per_changing_step():
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry(); reg.discover_builtin()
    ex = PipelineExecutor(reg)
    rng = np.random.default_rng(4)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    # sharpen пост-шаг меняет картинку -> хотя бы один вариант
    res = ex.execute(img, {"post": {"sharpen": 0.8}}, {})
    assert "variants" in res
    assert isinstance(res["variants"], list)
    assert all("label" in v and "image" in v for v in res["variants"])
    assert any("sharpen" in v["label"].lower() or v["label"]
               for v in res["variants"])  # непустые лейблы


def test_variants_collected_without_blend():
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    reg = PluginRegistry(); reg.discover_builtin()
    ex = PipelineExecutor(reg)
    # ПРИМЕЧАНИЕ (отклонение от brief): исходный тест использовал плоское
    # (128,128,128) изображение — unsharp mask математически не меняет
    # константное изображение (blur плоской картинки равен ей самой), поэтому
    # ни один вариант не мог бы появиться независимо от реализации.
    # Используем изображение с реальным содержимым, как в тесте выше.
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    res = ex.execute(img, {"post": {"sharpen": 0.5}}, {})  # blend не включён
    assert res.get("variants")  # варианты собраны даже без blend


def test_no_changing_step_yields_empty_variants():
    """Регресс (флаг задачи 3): пустой конфиг ничего не меняет -> варианты
    отсутствуют, но выполнение не падает."""
    import numpy as np
    from upscaler.plugins.registry import PluginRegistry
    ex = PipelineExecutor(PluginRegistry())
    img = np.full((32, 32, 3), 128, np.uint8)
    res = ex.execute(img, {}, {})   # empty config: nothing changes the image
    assert res.get("variants") == []
