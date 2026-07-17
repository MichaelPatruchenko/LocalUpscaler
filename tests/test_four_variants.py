"""Планировщик «4 варианта обработки» + стилевые директивы LLM-советника."""
import numpy as np
import pytest

from upscaler.engine.four_variants import (
    VARIANT_DIRECTIONS, build_variants,
)


class _FakeConfigurator:
    """Стабильный базовый конфиг вместо полного AutoConfigurator."""

    def __init__(self, config=None):
        self._config = config or {
            "scale": 4,
            "enhance_only": False,
            "denoise": {"SCUNet": {"strength": 0.4, "tile_size": 512}},
            "adjust": {"Refocus": {"strength": 0.6}},
            "upscale": {"plugin": "Real-ESRGAN", "scale": 4},
            "post": {"sharpen": 0.2},
        }
        self.calls = []

    def recommend(self, analysis, scale=4, enhance_only=False,
                  allow_predownscale=True):
        self.calls.append((scale, enhance_only, allow_predownscale))
        import copy
        return copy.deepcopy(self._config)


def test_directions_are_four_with_unique_ids():
    ids = [d["id"] for d in VARIANT_DIRECTIONS]
    assert ids == ["natural", "sharp", "clean", "vivid"]
    assert all(d["style_directive"] for d in VARIANT_DIRECTIONS)
    assert all(d["name_key"].startswith("variants.") for d in VARIANT_DIRECTIONS)


def test_build_variants_returns_four_wrappers():
    variants = build_variants({}, configurator=_FakeConfigurator())
    assert [v["id"] for v in variants] == ["natural", "sharp", "clean", "vivid"]
    for v in variants:
        assert set(v) == {"id", "name_key", "style_directive", "config"}
        # рабочий конфиг без посторонних ключей — уходит в worker как есть
        assert "variant_id" not in v["config"]


def test_build_variants_configs_are_independent():
    variants = build_variants({}, configurator=_FakeConfigurator())
    variants[0]["config"]["post"]["sharpen"] = 0.99
    assert variants[1]["config"]["post"]["sharpen"] != 0.99


def test_natural_keeps_base_config():
    fake = _FakeConfigurator()
    natural = build_variants({}, configurator=fake)[0]
    assert natural["config"] == fake._config


def test_sharp_raises_detail_and_softens_denoise():
    fake = _FakeConfigurator()
    base = fake._config
    sharp = build_variants({}, configurator=fake)[1]["config"]
    assert sharp["adjust"]["Refocus"]["strength"] > base["adjust"]["Refocus"]["strength"]
    assert sharp["post"]["sharpen"] > base["post"]["sharpen"]
    assert (sharp["denoise"]["SCUNet"]["strength"]
            < base["denoise"]["SCUNet"]["strength"])
    assert sharp["upscale"]["plugin"] == "HAT-S"  # scale=4 -> HAT-S


def test_sharp_keeps_upscaler_when_scale_not_4():
    cfg = {
        "scale": 8, "enhance_only": False, "denoise": {},
        "adjust": {}, "upscale": {"plugin": "Real-ESRGAN", "scale": 8},
        "post": {"sharpen": 0.2},
    }
    sharp = build_variants({}, scale=8,
                           configurator=_FakeConfigurator(cfg))[1]["config"]
    assert sharp["upscale"]["plugin"] == "Real-ESRGAN"


def test_clean_raises_denoise_and_softens_sharpening():
    fake = _FakeConfigurator()
    base = fake._config
    clean = build_variants({}, configurator=fake)[2]["config"]
    assert (clean["denoise"]["SCUNet"]["strength"]
            > base["denoise"]["SCUNet"]["strength"])
    assert clean["post"]["sharpen"] < base["post"]["sharpen"]
    assert (clean["adjust"]["Refocus"]["strength"]
            < base["adjust"]["Refocus"]["strength"])


def test_clean_adds_mild_denoise_when_none():
    cfg = {"scale": 4, "enhance_only": False, "denoise": {}, "adjust": {},
           "upscale": {}, "post": {"sharpen": 0.2}}
    clean = build_variants({}, configurator=_FakeConfigurator(cfg))[2]["config"]
    assert "NL-Means" in clean["denoise"]


def test_vivid_adds_saturation_and_contrast():
    vivid = build_variants({}, configurator=_FakeConfigurator())[3]["config"]
    assert vivid["adjust"]["Vibrance"]["strength"] > 0
    assert vivid["adjust"]["Auto Contrast"]["strength"] > 0


def test_values_stay_in_valid_ranges():
    cfg = {
        "scale": 4, "enhance_only": False,
        "denoise": {"SCUNet": {"strength": 0.9}},
        "adjust": {"Refocus": {"strength": 0.95},
                   "Vibrance": {"strength": 0.95}},
        "upscale": {"plugin": "Real-ESRGAN", "scale": 4},
        "post": {"sharpen": 0.95},
    }
    for v in build_variants({}, configurator=_FakeConfigurator(cfg)):
        c = v["config"]
        assert 0.0 <= c["post"]["sharpen"] <= 1.0
        for params in c["denoise"].values():
            if "strength" in params:
                assert 0.0 <= params["strength"] <= 1.0
        for params in c["adjust"].values():
            if "strength" in params:
                assert 0.0 <= params["strength"] <= 1.0


def test_recommend_receives_flags():
    fake = _FakeConfigurator()
    build_variants({}, scale=2, enhance_only=True, configurator=fake,
                   allow_predownscale=False)
    assert fake.calls == [(2, True, False)]


# ─── style_directive в промптах советника ───

def test_build_prompt_includes_directive_when_set():
    from upscaler.engine.llm_advisor import build_prompt
    directive = VARIANT_DIRECTIONS[1]["style_directive"]
    prompt = build_prompt("summary", {}, style_directive=directive)
    assert directive in prompt
    assert "STYLE DIRECTIVE" in prompt


def test_build_prompt_unchanged_when_directive_empty():
    from upscaler.engine.llm_advisor import build_prompt
    assert build_prompt("summary", {}) == build_prompt(
        "summary", {}, style_directive="")
    assert "STYLE DIRECTIVE" not in build_prompt("summary", {})


def test_build_evaluation_prompt_includes_directive_when_set():
    from upscaler.engine.llm_advisor import build_evaluation_prompt
    directive = VARIANT_DIRECTIONS[2]["style_directive"]
    prompt = build_evaluation_prompt("summary", style_directive=directive)
    assert directive in prompt


def test_build_evaluation_prompt_unchanged_when_directive_empty():
    from upscaler.engine.llm_advisor import build_evaluation_prompt
    assert build_evaluation_prompt("summary") == build_evaluation_prompt(
        "summary", style_directive="")


def test_refine_passes_directive_into_prompt():
    from upscaler.engine.llm_advisor import LLMAdvisor
    seen = {}

    def fake_generate(prompt, image):
        seen["prompt"] = prompt
        return "{}"

    advisor = LLMAdvisor(models_dir="nonexistent")
    advisor.refine(None, {}, {"scale": 4}, generate=fake_generate,
                   style_directive="Style priority: TEST-DIRECTIVE.")
    assert "TEST-DIRECTIVE" in seen["prompt"]


def test_evaluate_passes_directive_into_prompt():
    from upscaler.engine.llm_advisor import LLMAdvisor
    seen = {}

    def fake_generate(prompt, image):
        seen["prompt"] = prompt
        return '{"satisfied": true}'

    advisor = LLMAdvisor(models_dir="nonexistent")
    advisor.evaluate(None, {}, generate=fake_generate,
                     style_directive="Style priority: TEST-DIRECTIVE.")
    assert "TEST-DIRECTIVE" in seen["prompt"]
