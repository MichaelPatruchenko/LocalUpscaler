import json

import numpy as np

from upscaler.engine.llm_advisor import (
    build_analysis_summary, build_prompt, extract_json,
    validate_overrides, apply_overrides, LLMAdvisor,
)


_ANALYSIS = {
    "resolution": (800, 600), "megapixels": 0.48, "is_grayscale": False,
    "noise_level": 12.0, "blur_score": 60.0, "brightness": 90.0,
    "contrast": 28.0, "dynamic_range": 200.0, "detail_level": 15.0,
    "edge_density": 0.05, "saturation_mean": 35.0, "color_temperature": 1.1,
    "hist_skewness": 0.5, "glcm_homogeneity": 0.7,
}
_BASE = {
    "scale": 4, "enhance_only": False,
    "denoise": {"NL-Means": {"h": 12}}, "adjust": {"Auto Tone": {"strength": 0.2}},
    "upscale": {"plugin": "HAT-S", "scale": 4}, "post": {"sharpen": 0.2},
}


def test_summary_contains_key_metrics():
    s = build_analysis_summary(_ANALYSIS)
    assert "noise_level" in s and "blur_score" in s and "800x600" in s


def test_prompt_includes_base_and_schema():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE)
    assert "HAT-S" in p and "JSON" in p and "scale" in p


def test_prompt_omits_deblur_when_not_allowed():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_deblur=False)
    assert "blur_type" not in p
    assert "never include" in p.lower()


def test_prompt_includes_deblur_when_allowed():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_deblur=True)
    assert "blur_type" in p
    assert "motion" in p


def test_validate_drops_deblur_when_not_allowed():
    raw = {"sharpen": 0.4, "deblur": {"enabled": True, "method": "tv"}}
    out = validate_overrides(raw, allow_deblur=False)
    assert "deblur" not in out
    assert out["sharpen"] == 0.4


def test_refine_strips_deblur_when_not_allowed():
    advisor = LLMAdvisor(models_dir=".")

    def fake(prompt, image):
        # model still tries to return deblur; it must be ignored
        assert "blur_type" not in prompt
        return '{"deblur": {"enabled": true, "method": "tv"}, "sharpen": 0.5}'

    out = advisor.refine(None, _ANALYSIS, _BASE, generate=fake,
                         allow_deblur=False)
    assert out["post"]["sharpen"] == 0.5
    assert not out.get("deblur")


def test_prompt_mentions_order_and_steps():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_deblur=True)
    assert '"order"' in p
    assert "PROCESSING ORDER" in p
    assert "upscale" in p and "sharpen" in p


def test_validate_order_keeps_known_tokens_in_sequence():
    raw = {"order": ["sharpen", "upscale", "denoise", "bogus", "upscale"]}
    out = validate_overrides(raw)
    assert out["order"] == ["sharpen", "upscale", "denoise"]


def test_validate_order_accepts_granular_adjuster_tokens():
    out = validate_overrides(
        {"order": ["auto_color", "saturation", "brightness", "upscale", "sharpen"]})
    assert out["order"] == ["auto_color", "saturation", "brightness",
                            "upscale", "sharpen"]


def test_validate_order_drops_deblur_when_not_allowed():
    raw = {"order": ["denoise", "deblur", "upscale"]}
    out = validate_overrides(raw, allow_deblur=False)
    assert out["order"] == ["denoise", "upscale"]


def test_validate_order_empty_when_no_known_tokens():
    assert "order" not in validate_overrides({"order": ["bogus", 5]})
    assert "order" not in validate_overrides({"order": "not a list"})


def test_apply_overrides_sets_order():
    cfg = apply_overrides(_BASE, {"order": ["upscale", "sharpen"]})
    assert cfg["order"] == ["upscale", "sharpen"]


def test_refine_applies_order():
    advisor = LLMAdvisor(models_dir=".")
    out = advisor.refine(
        None, _ANALYSIS, _BASE,
        generate=lambda p, i: '{"order": ["denoise", "upscale", "sharpen"]}')
    assert out["order"] == ["denoise", "upscale", "sharpen"]


def test_extract_json_plain():
    assert extract_json('{"scale": 4}') == {"scale": 4}


def test_extract_json_fenced_with_prose():
    text = 'Here you go:\n```json\n{"sharpen": 0.5, "scale": 2}\n```\nDone.'
    assert extract_json(text) == {"sharpen": 0.5, "scale": 2}


def test_extract_json_garbage_returns_empty():
    assert extract_json("no json here") == {}
    assert extract_json("") == {}


def test_validate_clamps_and_whitelists():
    raw = {
        "scale": 3,                  # invalid -> dropped
        "enhance_only": True,
        "upscaler": "NotAModel",     # unknown -> dropped
        "sharpen": 5.0,              # clamp -> 1.0
        "saturation": -2.0,          # clamp -> -1.0
        "denoise_strength": 0.3,
        "deblur": {"enabled": True, "method": "rl", "blur_type": "bogus"},
        "unknown_key": 123,          # dropped
    }
    out = validate_overrides(raw)
    assert "scale" not in out
    assert out["enhance_only"] is True
    assert "upscaler" not in out
    assert out["sharpen"] == 1.0
    assert out["saturation"] == -1.0
    assert out["denoise_strength"] == 0.3
    assert out["deblur"] == {"enabled": True, "method": "rl"}
    assert "unknown_key" not in out


def test_apply_overrides_merges():
    overrides = {
        "scale": 2, "upscaler": "Real-ESRGAN", "sharpen": 0.4,
        "saturation": 0.3, "denoise_strength": 0.0,
        "deblur": {"enabled": True, "method": "tv"},
    }
    cfg = apply_overrides(_BASE, overrides)
    assert cfg["scale"] == 2
    assert cfg["upscale"] == {"plugin": "Real-ESRGAN", "scale": 2}
    assert cfg["post"]["sharpen"] == 0.4
    assert cfg["adjust"]["Saturation"]["direction"] == "up"
    assert cfg["denoise"] == {}            # strength 0 disables denoise
    assert cfg["deblur"]["method"] == "tv"
    # base config not mutated
    assert _BASE["scale"] == 4


def test_apply_overrides_negative_direction():
    cfg = apply_overrides(_BASE, {"brightness": -0.5})
    assert cfg["adjust"]["Brightness"] == {"strength": 0.5, "direction": "down"}


def test_refine_with_fake_generate():
    advisor = LLMAdvisor(models_dir=".")
    def fake(prompt, image):
        return '```json\n{"sharpen": 0.6, "scale": 2}\n```'
    out = advisor.refine(None, _ANALYSIS, _BASE, generate=fake)
    assert out["post"]["sharpen"] == 0.6
    assert out["scale"] == 2


def test_refine_falls_back_on_generate_error():
    advisor = LLMAdvisor(models_dir=".")
    def boom(prompt, image):
        raise RuntimeError("model crashed")
    out = advisor.refine(None, _ANALYSIS, _BASE, generate=boom)
    assert out == _BASE


def test_refine_falls_back_on_empty_overrides():
    advisor = LLMAdvisor(models_dir=".")
    out = advisor.refine(None, _ANALYSIS, _BASE, generate=lambda p, i: "no json")
    assert out == _BASE


def test_refine_unavailable_returns_base(tmp_path):
    # empty models dir -> no model -> not available -> base returned
    advisor = LLMAdvisor(models_dir=str(tmp_path))
    assert advisor.available() is False
    out = advisor.refine(None, _ANALYSIS, _BASE)
    assert out == _BASE


def test_mmproj_pairing_by_family(tmp_path):
    (tmp_path / "gemma-4-E2B-it-Q5_K_M.gguf").write_bytes(b"x")
    (tmp_path / "Qwen3.5-4B-Q5_K_M.gguf").write_bytes(b"x")
    (tmp_path / "gemma4mmproj-F16.gguf").write_bytes(b"x")
    (tmp_path / "qwen35mmproj-F16.gguf").write_bytes(b"x")
    # gemma model selected -> gemma mmproj
    a = LLMAdvisor(models_dir=str(tmp_path),
                   model_filename="gemma-4-E2B-it-Q5_K_M.gguf")
    assert "gemma" in a._mmproj_path().name.lower()
    # qwen model selected -> qwen mmproj
    b = LLMAdvisor(models_dir=str(tmp_path),
                   model_filename="Qwen3.5-4B-Q5_K_M.gguf")
    assert "qwen" in b._mmproj_path().name.lower()


def test_validate_full_deblur_params():
    raw = {"deblur": {
        "enabled": True, "blur_type": "motion", "radius": 8.0, "angle": 200.0,
        "smooth": 40, "edge_feather": 15, "correction_strength": 50,
        "method": "tv", "tv_iterations": 5000, "edge_taper": True,
    }}
    out = validate_overrides(raw)
    d = out["deblur"]
    assert d["blur_type"] == "motion"
    assert d["radius"] == 8.0
    assert d["angle"] == 180.0          # clamped to max
    assert d["smooth"] == 40
    assert d["tv_iterations"] == 1000   # clamped to max
    assert d["method"] == "tv"
    assert d["edge_taper"] is True


def test_apply_full_deblur_sets_manual_mode():
    overrides = {"deblur": {
        "enabled": True, "blur_type": "motion", "radius": 6.0, "angle": 45.0,
        "smooth": 30, "method": "rl",
    }}
    cfg = apply_overrides(_BASE, overrides)
    d = cfg["deblur"]
    assert d["auto"] is False           # explicit shape params -> manual
    assert d["blur_type"] == "motion"
    assert d["radius"] == 6.0 and d["angle"] == 45.0
    assert d["method"] == "rl"


def test_apply_deblur_auto_when_no_shape_params():
    cfg = apply_overrides(_BASE, {"deblur": {"enabled": True, "method": "wiener"}})
    assert cfg["deblur"]["auto"] is True
    assert cfg["deblur"]["method"] == "wiener"


def test_apply_denoiser_selection_and_auto_adjusters():
    overrides = {"denoiser": "SCUNet", "denoise_strength": 0.4,
                 "auto_tone": 0.5, "auto_contrast": 0.0, "refocus": 0.3}
    cfg = apply_overrides(_BASE, overrides)
    assert cfg["denoise"] == {"SCUNet": {"strength": 0.4}}
    assert cfg["adjust"]["Auto Tone"] == {"strength": 0.5}
    assert "Auto Contrast" not in cfg["adjust"]      # 0 -> not added
    assert cfg["post"]["refocus"] == {"strength": 0.3}


def test_encode_image_data_uri_roundtrip():
    img = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint8)
    uri = LLMAdvisor._encode_image_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    import base64
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_prompt_includes_icedit_when_allowed():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_icedit=True)
    assert "icedit" in p.lower()
    assert "instruction" in p.lower()


def test_prompt_omits_icedit_when_not_allowed():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_icedit=False)
    assert "instruction" not in p.lower()
    assert "icedit" not in p.lower()


def test_validate_keeps_icedit_when_allowed():
    raw = {"icedit": {"enabled": True,
                      "instruction": "  remove the watermark  ",
                      "variant": "moe"}}
    out = validate_overrides(raw, allow_icedit=True)
    assert out["icedit"]["instruction"] == "remove the watermark"
    assert out["icedit"]["variant"] == "moe"
    assert out["icedit"]["enabled"] is True


def test_validate_clamps_instruction_length():
    raw = {"icedit": {"enabled": True, "instruction": "x" * 500}}
    out = validate_overrides(raw, allow_icedit=True)
    assert len(out["icedit"]["instruction"]) == 300


def test_validate_drops_empty_instruction():
    raw = {"icedit": {"enabled": True, "instruction": "   "}}
    out = validate_overrides(raw, allow_icedit=True)
    assert "icedit" not in out


def test_validate_bad_variant_falls_back_to_normal():
    raw = {"icedit": {"enabled": True, "instruction": "edit", "variant": "xyz"}}
    out = validate_overrides(raw, allow_icedit=True)
    assert out["icedit"]["variant"] == "normal"


def test_validate_drops_icedit_when_not_allowed():
    raw = {"sharpen": 0.3,
           "icedit": {"enabled": True, "instruction": "edit"},
           "order": ["denoise", "icedit", "upscale"]}
    out = validate_overrides(raw, allow_icedit=False)
    assert "icedit" not in out
    assert "icedit" not in out.get("order", [])
    assert out["sharpen"] == 0.3


def test_apply_overrides_merges_icedit():
    cfg = apply_overrides(_BASE, {"icedit": {"enabled": True,
                                             "instruction": "edit", "variant": "moe"}})
    assert cfg["icedit"]["instruction"] == "edit"


def test_apply_overrides_icedit_disabled_clears():
    base = dict(_BASE, icedit={"instruction": "old", "variant": "moe"})
    cfg = apply_overrides(base, {"icedit": {"enabled": False}})
    assert cfg["icedit"] == {}


from upscaler.engine.llm_advisor import (
    should_continue_refinement, build_evaluation_prompt, validate_evaluation,
)


def test_should_continue_refinement_table():
    assert should_continue_refinement(1, 3, False) is True
    assert should_continue_refinement(2, 3, False) is True
    assert should_continue_refinement(3, 3, False) is False   # max reached
    assert should_continue_refinement(1, 3, True) is False     # satisfied
    assert should_continue_refinement(1, 1, False) is False     # max==1


def test_evaluation_prompt_has_satisfied_and_no_upscale():
    s = build_analysis_summary(_ANALYSIS)
    p = build_evaluation_prompt(s)
    assert "satisfied" in p.lower()
    assert "upscaler" not in p.lower()
    assert '"scale"' not in p


def test_evaluation_prompt_respects_allow_flags():
    s = build_analysis_summary(_ANALYSIS)
    p_off = build_evaluation_prompt(s, allow_deblur=False, allow_icedit=False)
    assert "blur_type" not in p_off
    assert "instruction" not in p_off.lower()


def test_validate_evaluation_reads_satisfied_and_strips_upscale():
    raw = {"satisfied": False, "sharpen": 0.3, "scale": 8,
           "upscaler": "HAT-S", "enhance_only": False,
           "order": ["denoise", "upscale", "sharpen"]}
    satisfied, overrides = validate_evaluation(raw)
    assert satisfied is False
    assert overrides["sharpen"] == 0.3
    assert "scale" not in overrides
    assert "upscaler" not in overrides
    assert "enhance_only" not in overrides
    assert "upscale" not in overrides.get("order", [])


def test_validate_evaluation_defaults_satisfied_true():
    satisfied, overrides = validate_evaluation({"sharpen": 0.2})
    assert satisfied is True


def test_evaluate_satisfied_returns_no_config():
    advisor = LLMAdvisor(models_dir=".")

    def fake(prompt, image):
        return '{"satisfied": true}'

    out = advisor.evaluate(None, _ANALYSIS, generate=fake)
    assert out["satisfied"] is True
    assert out["config"] is None


def test_evaluate_unsatisfied_returns_enhance_only_config():
    advisor = LLMAdvisor(models_dir=".")

    def fake(prompt, image):
        return ('{"satisfied": false, "sharpen": 0.3, "scale": 8, '
                '"upscaler": "HAT-S"}')

    out = advisor.evaluate(None, _ANALYSIS, generate=fake)
    assert out["satisfied"] is False
    cfg = out["config"]
    assert cfg["enhance_only"] is True
    assert cfg.get("post", {}).get("sharpen") == 0.3
    # the requested upscale was stripped — no upscaler plugin
    assert not cfg.get("upscale", {}).get("plugin")


def test_evaluate_unavailable_advisor_stops():
    advisor = LLMAdvisor(models_dir=".")  # no real model present
    out = advisor.evaluate(None, _ANALYSIS)  # generate=None, unavailable
    assert out["satisfied"] is True
    assert out["config"] is None


from upscaler.engine.llm_advisor import _validate_face


def test_validate_face_clamps_fidelity():
    out = _validate_face({"enabled": True, "fidelity": 5.0,
                          "upscale_background": True})
    assert out["enabled"] is True
    assert out["fidelity"] == 1.0
    assert out["upscale_background"] is True


def test_validate_face_rejects_non_dict():
    assert _validate_face(None) == {}
    assert _validate_face("x") == {}


def test_allow_face_true_keeps_block_and_token():
    raw = {"face": {"enabled": True, "fidelity": 0.6},
           "order": ["upscale", "face_restore", "sharpen"]}
    out = validate_overrides(raw, allow_face=True)
    assert out["face"]["fidelity"] == 0.6
    assert "face_restore" in out["order"]


def test_allow_face_false_drops_block_and_token():
    raw = {"face": {"enabled": True, "fidelity": 0.6},
           "order": ["upscale", "face_restore", "sharpen"]}
    out = validate_overrides(raw, allow_face=False)
    assert "face" not in out
    assert "face_restore" not in out.get("order", [])


def test_apply_overrides_merges_face_and_disable():
    base = {"face": {"enabled": True, "fidelity": 0.7}}
    on = apply_overrides(base, {"face": {"fidelity": 0.4}})
    assert on["face"]["fidelity"] == 0.4
    off = apply_overrides(base, {"face": {"enabled": False}})
    assert off["face"] == {}


def test_prompt_warns_against_oversoftening():
    p = build_prompt("noise_level: 5", {"scale": 4})
    assert "avoid over-softening" in p.lower()


def test_evaluation_prompt_warns_against_oversoftening():
    p = build_evaluation_prompt("noise_level: 5")
    assert "avoid over-softening" in p.lower()


def test_advisor_default_n_gpu_layers_offloads_all(tmp_path):
    # Default is -1: "offload all layers when the llama-cpp build supports CUDA"
    # (a CPU-only build ignores it). This is the use-GPU-when-possible default.
    adv = LLMAdvisor(models_dir=tmp_path)
    assert adv.n_gpu_layers == -1


def test_advisor_passes_n_gpu_layers_to_llama(tmp_path, monkeypatch):
    import sys
    import types

    # A fake model file so model_path() resolves; no mmproj -> text-only path.
    (tmp_path / "model.gguf").write_bytes(b"gguf")

    captured = {}

    fake_mod = types.ModuleType("llama_cpp")

    class _FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_mod.Llama = _FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)

    adv = LLMAdvisor(models_dir=tmp_path, n_gpu_layers=7)
    adv._build_llm()
    assert captured["n_gpu_layers"] == 7


def test_vision_handler_prefers_available_family_handler(monkeypatch, tmp_path):
    """The family handler must use names present in the installed llama-cpp.
    Regression: 0.3.31 renamed Gemma3->Gemma4 and dropped Qwen2VL; the advisor
    must not silently fall back to Llava for a gemma/qwen model when a real
    family handler exists."""
    from upscaler.engine.llm_advisor import LLMAdvisor
    from llama_cpp import llama_chat_format as fmt

    class _Rec:
        def __init__(self, clip_model_path=None):
            self.clip_model_path = clip_model_path

    class _Gemma4(_Rec):
        pass

    class _Qwen25(_Rec):
        pass

    class _Llava(_Rec):
        pass

    # Simulate the 0.3.31 surface: Gemma4 + Qwen25VL + Llava present; the old
    # Gemma3/Qwen2VL names absent.
    monkeypatch.setattr(fmt, "Gemma4ChatHandler", _Gemma4, raising=False)
    monkeypatch.setattr(fmt, "Qwen25VLChatHandler", _Qwen25, raising=False)
    monkeypatch.setattr(fmt, "Llava15ChatHandler", _Llava, raising=False)
    monkeypatch.delattr(fmt, "Gemma3ChatHandler", raising=False)
    monkeypatch.delattr(fmt, "Qwen2VLChatHandler", raising=False)

    adv = LLMAdvisor(models_dir=tmp_path)

    monkeypatch.setattr(adv, "_model_family", lambda *a, **k: "gemma")
    assert isinstance(adv._vision_handler(tmp_path / "mm.gguf"), _Gemma4)

    monkeypatch.setattr(adv, "_model_family", lambda *a, **k: "qwen")
    assert isinstance(adv._vision_handler(tmp_path / "mm.gguf"), _Qwen25)


def test_validate_icedit_variant_fallback_is_normal():
    from upscaler.engine.llm_advisor import _validate_icedit
    out = _validate_icedit({"enabled": True, "instruction": "x",
                            "variant": "bogus"})
    assert out.get("variant") == "normal"


# --- Этап 6A: осведомлённость советника о смешивании вариантов ---------------

def test_refine_prompt_mentions_blend_when_enabled():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_deblur=False, allow_icedit=False,
                     allow_face=False, blend_enabled=True)
    assert "смешивание вариантов" in p.lower() or "blend" in p.lower()


def test_refine_prompt_omits_blend_when_disabled():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE, allow_deblur=False, allow_icedit=False,
                     allow_face=False, blend_enabled=False)
    assert "blend" not in p.lower()


def test_refine_prompt_blend_defaults_to_disabled():
    s = build_analysis_summary(_ANALYSIS)
    p = build_prompt(s, _BASE)
    assert "blend" not in p.lower()


def test_validate_overrides_drops_unknown_blend_key():
    raw = {"blend": {"enabled": True}, "scale": 4}
    out = validate_overrides(raw)
    assert "blend" not in out
    assert out["scale"] == 4


# --- Этап B: советник знает новые корректоры ----------------------------------


def test_validate_accepts_new_scalar_adjusters():
    out = validate_overrides({
        "auto_levels": 0.5, "clarity": 0.3, "dehaze": 0.6,
        "white_balance": 0.7, "dodge_burn": 0.2, "skin_smooth": 0.4,
        "vibrance": -0.5, "shadows_lift": 0.4, "highlights_tame": 0.2,
    })
    assert out["auto_levels"] == 0.5
    assert out["vibrance"] == -0.5
    assert out["shadows_lift"] == 0.4


def test_validate_clamps_new_adjusters():
    out = validate_overrides({"dehaze": 5.0, "vibrance": -3.0})
    assert out["dehaze"] <= 1.0
    assert out["vibrance"] >= -1.0


def test_validate_optics_and_split_tone_dicts():
    out = validate_overrides({
        "optics": {"vignette": 0.5, "ca": 0.3},
        "split_tone": {"shadow_hue": 500.0, "highlight_hue": 45.0,
                       "saturation": 0.4, "balance": 0.1},
    })
    assert out["optics"]["vignette"] == 0.5
    assert 0.0 <= out["split_tone"]["shadow_hue"] <= 360.0


def test_apply_overrides_new_adjusters():
    cfg = apply_overrides({"adjust": {}}, {
        "dehaze": 0.6, "vibrance": -0.4,
        "shadows_lift": 0.4, "highlights_tame": 0.2,
        "optics": {"vignette": 0.5, "ca": 0.3},
    })
    assert cfg["adjust"]["Dehaze"] == {"strength": 0.6}
    assert cfg["adjust"]["Vibrance"] == {"strength": -0.4}
    assert cfg["adjust"]["Shadows/Highlights"]["shadows"] == 0.4
    assert cfg["adjust"]["Optics"]["vignette"] == 0.5


def test_apply_overrides_zero_removes_new_adjuster():
    base = {"adjust": {"Dehaze": {"strength": 0.5}}}
    cfg = apply_overrides(base, {"dehaze": 0.0})
    assert "Dehaze" not in cfg["adjust"]


def test_partial_shadows_override_preserves_algorithmic_highlights():
    from upscaler.engine.llm_advisor import apply_overrides
    base = {"adjust": {"Shadows/Highlights": {"shadows": 0.3,
                                              "highlights": 0.35,
                                              "radius": 30}}}
    cfg = apply_overrides(base, {"shadows_lift": 0.5})
    sh = cfg["adjust"]["Shadows/Highlights"]
    assert sh["shadows"] == 0.5
    assert sh["highlights"] == 0.35


def test_order_accepts_new_tokens():
    out = validate_overrides({"order": ["dehaze", "clarity", "upscale",
                                        "bogus_token"]})
    assert "dehaze" in out.get("order", [])
    assert "bogus_token" not in out.get("order", [])


def test_prompt_mentions_new_adjusters():
    # build_prompt's real signature is (summary: str, base_config: dict, ...);
    # adapted from the brief's template (which passed {} for summary) to match.
    from upscaler.engine import llm_advisor as la
    prompt = la.build_prompt("", {}, allow_deblur=False, allow_icedit=False,
                             allow_face=False)
    for key in ("dehaze", "vibrance", "white_balance", "shadows_lift",
                "clarity", "auto_levels"):
        assert key in prompt, key
