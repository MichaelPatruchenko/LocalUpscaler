"""Четыре направления автоматической обработки («4 варианта обработки»).

Из одного результата SourceAnalyzer строятся 4 конфига: базовая рекомендация
AutoConfigurator + умеренное смещение направления. Направления объявлены как
данные (VARIANT_DIRECTIONS) — единый источник для планировщика, оркестратора
и галереи выбора.

Не путать с history/variant_store.py: там «варианты» — снимки шагов конвейера
для вкладки «Смешивание». Здесь — четыре альтернативных конфига обработки.
"""
import copy
import logging

log = logging.getLogger(__name__)

# style_directive — по-английски: промпты LLM-советника англоязычные.
VARIANT_DIRECTIONS = [
    {
        "id": "natural",
        "name_key": "variants.natural",
        "style_directive": (
            "Style priority: NATURAL and faithful. Keep the result balanced "
            "and true to the source; avoid aggressive sharpening, saturation "
            "or heavy denoising."),
    },
    {
        "id": "sharp",
        "name_key": "variants.sharp",
        "style_directive": (
            "Style priority: MAXIMUM sharpness and fine detail. Favor "
            "refocus/sharpening and detail-preserving choices; light noise "
            "is acceptable; avoid over-smoothing."),
    },
    {
        "id": "clean",
        "name_key": "variants.clean",
        "style_directive": (
            "Style priority: CLEAN, noise-free result. Favor stronger "
            "denoising and smooth surfaces; keep sharpening restrained to "
            "avoid re-amplifying noise."),
    },
    {
        "id": "vivid",
        "name_key": "variants.vivid",
        "style_directive": (
            "Style priority: VIVID, expressive look. Favor richer saturation/"
            "vibrance, contrast and tonal punch while keeping skin tones "
            "believable."),
    },
]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _scale_denoise(denoise: dict, factor: float) -> None:
    """Умеренно масштабирует силу уже выбранного денойзера (in-place)."""
    for name, params in denoise.items():
        if not isinstance(params, dict):
            continue
        if "strength" in params:
            params["strength"] = round(_clamp01(params["strength"] * factor), 2)
        elif "h" in params:  # NL-Means: сила задаётся параметром h (1..20)
            params["h"] = int(max(1, min(20, round(params["h"] * factor))))


def _bias_sharp(cfg: dict) -> None:
    refocus = cfg["adjust"].get("Refocus")
    if refocus and "strength" in refocus:
        refocus["strength"] = round(_clamp01(refocus["strength"] + 0.15), 2)
    else:
        cfg["adjust"]["Refocus"] = {"strength": 0.4}
    cfg["post"]["sharpen"] = round(
        _clamp01(float(cfg["post"].get("sharpen", 0.0)) + 0.15), 2)
    _scale_denoise(cfg["denoise"], 0.6)
    # HAT-S — максимум детализации, но поддерживает только 4x.
    up = cfg.get("upscale") or {}
    if not cfg.get("enhance_only") and up.get("scale") == 4:
        up["plugin"] = "HAT-S"


def _bias_clean(cfg: dict) -> None:
    if cfg["denoise"]:
        _scale_denoise(cfg["denoise"], 1.4)
    else:
        # Источник без явного шума: мягкий NL-Means всё равно даёт «чистое»
        # направление, не убивая деталь.
        cfg["denoise"]["NL-Means"] = {"h": 5, "template_window": 7,
                                      "search_window": 21}
    refocus = cfg["adjust"].get("Refocus")
    if refocus and "strength" in refocus:
        refocus["strength"] = round(_clamp01(refocus["strength"] * 0.5), 2)
    cfg["post"]["sharpen"] = round(
        _clamp01(float(cfg["post"].get("sharpen", 0.0)) * 0.5), 2)


def _bias_vivid(cfg: dict) -> None:
    vib = cfg["adjust"].get("Vibrance")
    if vib and "strength" in vib:
        vib["strength"] = round(_clamp01(vib["strength"] + 0.2), 2)
    else:
        cfg["adjust"]["Vibrance"] = {"strength": 0.3}
    ac = cfg["adjust"].get("Auto Contrast")
    if ac and "strength" in ac:
        ac["strength"] = round(_clamp01(ac["strength"] + 0.1), 2)
    else:
        cfg["adjust"]["Auto Contrast"] = {"strength": 0.25}
    tone = cfg["adjust"].get("Auto Tone")
    if tone and "strength" in tone:
        tone["strength"] = round(_clamp01(tone["strength"] + 0.1), 2)


_BIASES = {
    "natural": lambda cfg: None,
    "sharp": _bias_sharp,
    "clean": _bias_clean,
    "vivid": _bias_vivid,
}


def build_variants(analysis: dict, scale: int = 4, enhance_only: bool = False,
                   configurator=None,
                   allow_predownscale: bool = True) -> list[dict]:
    """4 обёртки {id, name_key, style_directive, config} из одного анализа.

    Конфиги независимы (deep copy) и не содержат посторонних ключей — мета
    варианта живёт в обёртке, рабочий конфиг уходит в worker как есть.
    """
    if configurator is None:
        from upscaler.engine.auto_config import AutoConfigurator
        configurator = AutoConfigurator()
    base = configurator.recommend(analysis, scale=scale,
                                  enhance_only=enhance_only,
                                  allow_predownscale=allow_predownscale)
    out = []
    for direction in VARIANT_DIRECTIONS:
        cfg = copy.deepcopy(base)
        for key in ("denoise", "adjust", "upscale", "post"):
            cfg.setdefault(key, {})
        _BIASES[direction["id"]](cfg)
        out.append({
            "id": direction["id"],
            "name_key": direction["name_key"],
            "style_directive": direction["style_directive"],
            "config": cfg,
        })
        log.debug("Variant %s config built", direction["id"])
    return out
