"""LLM-based refinement of automatic processing parameters.

The auto buttons first compute an algorithmic pipeline config (AutoConfigurator).
This advisor optionally feeds that config plus a description of the image
(either the image itself, when a vision-capable GGUF runtime is available, or a
rich textual summary derived from SourceAnalyzer) to a local GGUF model and asks
it to return refined parameters as JSON. The refinement is validated, clamped to
known-safe ranges, and merged over the algorithmic config.

Everything degrades gracefully: if the runtime or model is unavailable, or the
model returns garbage, the original algorithmic config is returned unchanged.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

# Upscalers the model is allowed to choose from (validated against this set).
_KNOWN_UPSCALERS = {
    "Real-ESRGAN", "HAT-S", "SwinIR", "OmniSR", "DAT",
    "Lanczos", "Bicubic", "Sinc", "NEDI", "DCCI", "EGGI/SIRE",
}
_KNOWN_DEBLUR_METHODS = {"wiener", "tikhonov", "tv", "rl"}
_KNOWN_BLUR_TYPES = {"focus", "motion", "gaussian"}
_VALID_SCALES = {2, 4, 8, 16}
_KNOWN_DENOISERS = {"SCUNet", "NAFNet", "BM3D", "NL-Means", "Bilateral", "Wavelet"}
_KNOWN_ICEDIT_VARIANTS = {"moe", "normal"}
_ICEDIT_MAX_INSTRUCTION = 300

# Individual-operation tokens the model may sequence via the "order" key. Each
# adjuster token matches the override key of the same name, so one JSON answer
# selects a method, its parameters AND its position. Kept in sync with
# PipelineExecutor (the generic "adjust" fallback token is intentionally not
# exposed to the model). Not imported, to keep this module light.
_PIPELINE_STEPS = [
    "denoise", "deblur", "icedit", "optics",
    "white_balance", "dehaze",
    "auto_color", "auto_levels", "auto_tone", "auto_contrast",
    "brightness", "contrast", "shadows_highlights", "dodge_burn",
    "saturation", "vibrance", "split_tone", "clarity", "sharpness",
    "upscale", "face_restore", "skin_smooth",
    "refocus", "sharpen", "colorize",
]

# Numeric ranges for SmartDeblur parameters (mirror SmartDeblurPlugin.params_schema).
_DEBLUR_RANGES = {
    "radius": (0.1, 50.0),
    "angle": (0.0, 180.0),
    "smooth": (1.0, 100.0),
    "edge_feather": (0.0, 100.0),
    "correction_strength": (0.0, 100.0),
    "tv_iterations": (10, 1000),
}

# Preferred model filenames, in order. First match in the models dir is used.
_PREFERRED_MODELS = [
    "gemma-4-E2B-it-Q5_K_M.gguf",
    "Qwen3.5-4B-Q5_K_M.gguf",
    "gemma-4-E2B.Q5_K_M.gguf",
]


def build_analysis_summary(analysis: dict) -> str:
    """Render a compact, model-readable description of the image's metrics."""
    def g(key, default=0.0):
        return analysis.get(key, default)

    w, h = analysis.get("resolution", (0, 0))
    lines = [
        f"resolution: {w}x{h} ({g('megapixels', 0):.1f} MP)",
        f"grayscale: {bool(analysis.get('is_grayscale', False))}",
        f"noise_level: {g('noise_level'):.1f}",
        f"blur_score: {g('blur_score'):.0f} (lower = blurrier)",
        f"brightness: {g('brightness'):.0f}/255",
        f"contrast: {g('contrast'):.0f}",
        f"dynamic_range: {g('dynamic_range'):.0f}",
        f"detail_level: {g('detail_level'):.1f}",
        f"edge_density: {g('edge_density'):.3f}",
        f"saturation_mean: {g('saturation_mean'):.0f}",
        f"color_temperature: {g('color_temperature'):.2f}",
        f"hist_skewness: {g('hist_skewness'):.2f}",
        f"glcm_homogeneity: {g('glcm_homogeneity'):.2f}",
    ]
    return "\n".join(lines)


_DEBLUR_SCHEMA_BLOCK = (
    '  "deblur": {\n'
    '    "enabled": bool, "auto": bool,\n'
    '    "blur_type": "focus"|"motion"|"gaussian",\n'
    '    "radius": 0.1-50, "angle": 0-180, "smooth": 1-100,\n'
    '    "edge_feather": 0-100, "correction_strength": 0-100,\n'
    '    "method": "wiener"|"tikhonov"|"tv"|"rl",\n'
    '    "tv_iterations": 10-1000, "edge_taper": bool\n'
    '  }\n'
)

_DEBLUR_GUIDE = (
    "DEBLUR (deconvolution) - use ONLY for genuinely blurred sources "
    "(blur_score < ~120 or visibly soft):\n"
    "- Read the blur from the image and set blur_type:\n"
    "    * directional streaks/smear -> \"motion\"; set angle to the streak "
    "direction (0=horizontal, 90=vertical) and radius ~ half the streak "
    "length in pixels.\n"
    "    * uniformly soft / out-of-focus, no direction -> \"focus\"; radius ~ "
    "the blur circle radius (typically 2-8).\n"
    "    * mild general softness -> \"gaussian\".\n"
    "  Then set enabled=true with explicit blur_type/radius/angle and "
    "smooth 25-40. If you cannot read the blur confidently, set "
    "enabled=true, auto=true and let the algorithm estimate it.\n"
    "- method: \"wiener\" is the safe default; \"rl\" (Richardson-Lucy) for "
    "strong but clean blur; \"tv\" when the source is also noisy; "
    "\"tikhonov\" for the gentlest correction.\n"
    "- correction_strength 0 lets the method auto-balance; raise (20-60) only "
    "if residual blur remains. Keep edge_taper=true to avoid ringing at the "
    "borders. If the image is already sharp, omit deblur entirely.\n"
)

_ICEDIT_SCHEMA_BLOCK = (
    '  "icedit": {\n'
    '    "enabled": bool,\n'
    '    "instruction": "<short imperative edit, e.g. remove the watermark>",\n'
    '    "variant": "moe"|"normal"\n'
    '  }\n'
)

_ICEDIT_GUIDE = (
    "ICEDIT (instruction editing) - use ONLY for genuine SEMANTIC/content "
    "changes you can SEE are needed and can phrase as one short imperative "
    "(e.g. \"remove the watermark\", \"remove the blemish on the cheek\", "
    "\"make the hair green\", \"add sunglasses\"). It regenerates the region, "
    "so do NOT use it for pure restoration (denoise/sharpen/upscale/tone). "
    "Set enabled=true with a non-empty instruction; variant \"normal\" is the "
    "default; \"moe\" is experimental and should not be chosen. If no content "
    "change is clearly needed, OMIT icedit entirely.\n"
)

_FACE_SCHEMA_BLOCK = (
    '  "face": {\n'
    '    "enabled": bool, "fidelity": 0.0-1.0,\n'
    '    "upscale_background": bool\n'
    '  }\n'
)

_FACE_GUIDE = (
    "FACE RESTORATION (CodeFormer) - use when the image contains human "
    "FACES/portraits and they look soft, low-detail or degraded. It "
    "reconstructs realistic skin/eyes/hair on detected faces (this is what "
    "makes portraits look professionally restored). Set enabled=true with "
    "fidelity 0.6-0.8 (higher = closer to the original identity, lower = more "
    "aggressive reconstruction). Do NOT use it on images without faces. It runs "
    "AFTER upscaling, so keep face_restore after upscale in the order.\n"
)

_OPTICS_SCHEMA_BLOCK = (
    '  "optics": {\n'
    '    "vignette": -1.0-1.0, "ca": 0.0-1.0\n'
    '  }\n'
)

_SPLIT_TONE_SCHEMA_BLOCK = (
    '  "split_tone": {\n'
    '    "shadow_hue": 0-360, "highlight_hue": 0-360,\n'
    '    "saturation": 0.0-1.0, "balance": -1.0-1.0\n'
    '  }\n'
)

_NEW_ADJUSTERS_GUIDE = (
    "LOCAL/TONAL CORRECTORS (fine local/tonal adjustments; each 0.0-1.0 strength "
    "unless noted; vibrance is preferred over saturation for photos; use "
    "white_balance for strong color casts, auto_color for mild ones):\n"
    "- auto_levels: stretches/normalizes a narrow or clipped tonal range.\n"
    "- clarity: local midtone contrast (\"punch\"); 0.2-0.4 for hazy/flat "
    "detail; avoid on already-crisp portraits (adds texture/pores).\n"
    "- dehaze: removes atmospheric haze/fog on landscapes with a washed-out "
    "veil and low contrast.\n"
    "- white_balance: corrects a strong color cast (e.g. orange indoor "
    "light, blue shade); prefer auto_color for a mild cast.\n"
    "- dodge_burn: subtle shadow-lighten/highlight-darken for depth; light "
    "touch, 0.1-0.3.\n"
    "- skin_smooth: softens skin texture on portraits only (requires a "
    "detected face); never use on non-portrait images.\n"
    "- vibrance (-1.0-1.0): like saturation but protects skin tones and "
    "already-saturated colors; preferred over saturation for photos.\n"
    "- shadows_lift / highlights_tame (0.0-1.0 each): recover shadow/"
    "highlight detail without a global exposure change (combined into one "
    "\"Shadows/Highlights\" correction); use when dynamic_range is very "
    "high (blown highlights or blocked shadows).\n"
    "- optics {vignette -1.0-1.0, ca 0.0-1.0}: vignette>0 darkens the "
    "corners (<0 brightens them); ca corrects red/cyan or blue/yellow "
    "fringing at high-contrast edges.\n"
    "- split_tone {shadow_hue/highlight_hue 0-360, saturation 0.0-1.0, "
    "balance -1.0-1.0}: tints shadows and highlights with different hues "
    "for a stylized look; use sparingly, saturation 0.1-0.3.\n"
)

_BLEND_NOTE = (
    "Note: after your pipeline runs, an automatic blend step may compose "
    "intermediate variants (Photoshop blend modes) to maximize a quality "
    "score. Do not add sharpening/contrast steps solely to compensate - "
    "the blend step handles final polish.")

_ANTI_SOFTEN_RULE = (
    "CRITICAL - AVOID OVER-SOFTENING: the most common failure is a washed-out, "
    "blurry, low-detail result. Do NOT over-denoise; keep skin, hair and "
    "foliage texture. Use the minimum denoise that removes real noise. When "
    "unsure between cleaner and sharper, choose SHARPER. Preserve and recover "
    "fine detail; a slightly noisy but crisp result beats a smooth mushy one.\n"
)


def should_continue_refinement(iteration: int, max_iter: int,
                               satisfied: bool) -> bool:
    """Whether to run another refinement pass."""
    return (not satisfied) and iteration < max_iter


def _style_block(style_directive: str) -> str:
    """Render a non-empty style directive as a prompt priority block."""
    directive = (style_directive or "").strip()
    if not directive:
        return ""
    return (
        "STYLE DIRECTIVE (this run targets one of several styled variants; "
        "bias every parameter and ordering decision toward it):\n"
        + directive + "\n\n"
    )


def build_evaluation_prompt(summary: str, allow_deblur: bool = True,
                            allow_icedit: bool = True,
                            allow_face: bool = True,
                            style_directive: str = "") -> str:
    """Prompt asking the model to judge a PROCESSED result and, if needed,
    return refinement-only overrides (never upscale)."""
    schema = (
        '{\n'
        '  "satisfied": bool,\n'
        '  "denoiser": one of ' + str(sorted(_KNOWN_DENOISERS)) + ' or "none",\n'
        '  "denoise_strength": 0.0-1.0,\n'
        '  "auto_tone": 0.0-1.0, "auto_contrast": 0.0-1.0, "auto_color": 0.0-1.0,\n'
        '  "brightness": -1.0-1.0, "contrast": -1.0-1.0, "saturation": -1.0-1.0,\n'
        '  "sharpness": 0.0-1.0, "sharpen": 0.0-1.0, "refocus": 0.0-1.0'
    )
    refine_steps = [s for s in _PIPELINE_STEPS if s != "upscale"]
    if not allow_deblur:
        refine_steps = [s for s in refine_steps if s != "deblur"]
    if not allow_icedit:
        refine_steps = [s for s in refine_steps if s != "icedit"]
    if not allow_face:
        refine_steps = [s for s in refine_steps if s != "face_restore"]
    schema += ',\n  "order": ordered subset of ' + str(refine_steps)
    if allow_deblur:
        schema += ',\n' + _DEBLUR_SCHEMA_BLOCK
    if allow_icedit:
        schema += ',\n' + _ICEDIT_SCHEMA_BLOCK
    if allow_face:
        schema += ',\n' + _FACE_SCHEMA_BLOCK
    schema += '}\n'

    return (
        _ANTI_SOFTEN_RULE + "\n"
        + _style_block(style_directive) +
        "You are an expert photo-restoration critic. You are shown an ALREADY "
        "PROCESSED image and its measured properties. Judge whether it is a "
        "flawless final result.\n\n"
        "Measured properties (blur_score: lower = blurrier):\n" + summary + "\n\n"
        "If the image is already excellent, return {\"satisfied\": true} and "
        "nothing else. If it still has fixable problems (residual noise, "
        "softness, halos, color cast, banding, leftover artifacts, or a needed "
        "edit), return \"satisfied\": false together with ONLY refinement "
        "operations that fix the specific problem. NEVER upscale or rescale - "
        "the image is already at final resolution; do not include scale or any "
        "upscale model. Prefer restraint: mild under-correction beats new artifacts.\n\n"
        "Return ONLY a JSON object using these keys (omit unchanged keys):\n"
        + schema +
        "Respond with the JSON object only - no prose, no code fences."
    )


def validate_evaluation(raw: dict, allow_deblur: bool = True,
                        allow_icedit: bool = True,
                        allow_face: bool = True) -> tuple:
    """Return (satisfied, refinement_overrides) with upscale keys removed."""
    if not isinstance(raw, dict):
        return True, {}
    satisfied = bool(raw.get("satisfied", True))
    overrides = validate_overrides(raw, allow_deblur=allow_deblur,
                                   allow_icedit=allow_icedit,
                                   allow_face=allow_face)
    overrides.pop("scale", None)
    overrides.pop("upscaler", None)
    overrides.pop("enhance_only", None)
    if "order" in overrides:
        overrides["order"] = [t for t in overrides["order"] if t != "upscale"]
        if not overrides["order"]:
            overrides.pop("order")
    return satisfied, overrides


def build_prompt(summary: str, base_config: dict,
                 allow_deblur: bool = True,
                 allow_icedit: bool = True,
                 allow_face: bool = True,
                 blend_enabled: bool = False,
                 style_directive: str = "") -> str:
    """Build the instruction prompt asking for a refined-parameters JSON.

    When *allow_deblur* is False the deblur option is removed from both the
    schema and the guidance, and the model is told explicitly never to emit a
    ``deblur`` key (the user disabled SmartDeblur for automatic operations).
    When *allow_icedit* is False the icedit option is similarly removed.
    When *allow_face* is False the face restoration option is similarly removed.
    When *blend_enabled* is True the model is informed that a later automatic
    blend step may run (see ``_BLEND_NOTE``); the advisor never emits a
    ``blend`` key itself - blend is not part of the model's output schema.
    A non-empty *style_directive* (four-variants mode) is inserted as a
    priority block so the model biases its choices toward that direction;
    empty keeps the prompt byte-identical to the previous behavior.
    """
    schema = (
        '{\n'
        '  "scale": 2|4|8|16,\n'
        '  "enhance_only": bool,\n'
        '  "upscaler": one of ' + str(sorted(_KNOWN_UPSCALERS)) + ",\n"
        '  "denoiser": one of ' + str(sorted(_KNOWN_DENOISERS)) + ' or "none",\n'
        '  "denoise_strength": 0.0-1.0,\n'
        '  "auto_tone": 0.0-1.0, "auto_contrast": 0.0-1.0, "auto_color": 0.0-1.0,\n'
        '  "brightness": -1.0-1.0, "contrast": -1.0-1.0, "saturation": -1.0-1.0,\n'
        '  "sharpness": 0.0-1.0, "sharpen": 0.0-1.0, "refocus": 0.0-1.0,\n'
        '  "auto_levels": 0.0-1.0, "clarity": 0.0-1.0, "dehaze": 0.0-1.0,\n'
        '  "white_balance": 0.0-1.0, "dodge_burn": 0.0-1.0, '
        '"skin_smooth": 0.0-1.0,\n'
        '  "vibrance": -1.0-1.0, "shadows_lift": 0.0-1.0, '
        '"highlights_tame": 0.0-1.0'
    )
    order_steps = _PIPELINE_STEPS if allow_deblur else \
        [s for s in _PIPELINE_STEPS if s != "deblur"]
    if not allow_icedit:
        order_steps = [s for s in order_steps if s != "icedit"]
    if not allow_face:
        order_steps = [s for s in order_steps if s != "face_restore"]
    order_line = ',\n  "order": ordered subset of ' + str(order_steps)
    schema += order_line
    schema += ',\n' + _OPTICS_SCHEMA_BLOCK
    schema += ',\n' + _SPLIT_TONE_SCHEMA_BLOCK
    if allow_deblur:
        schema += ',\n' + _DEBLUR_SCHEMA_BLOCK
    if allow_icedit:
        schema += ',\n' + _ICEDIT_SCHEMA_BLOCK
    if allow_face:
        schema += ',\n' + _FACE_SCHEMA_BLOCK
    schema += '}\n'

    guide = (
        _ANTI_SOFTEN_RULE + "\n"
        "DECISION GUIDE (combine what you SEE with the metrics; the proposed "
        "config is already reasonable - override only when you are confident it "
        "improves THIS image, and prefer restraint: halos, plastic skin, "
        "banding and oversaturation look worse than mild under-processing):\n\n"
        "NOISE / GRAIN (noise_level):\n"
        "- <=6: leave denoising off to keep detail.\n"
        "- 6-12: light denoise - denoiser \"NL-Means\", denoise_strength 0.15-0.30.\n"
        "- >12 on smooth areas (glcm_homogeneity>0.6): denoiser \"SCUNet\", "
        "strength 0.4-0.8.\n"
        "- >12 on textured areas (glcm_homogeneity<=0.6): \"SCUNet\" gentle "
        "0.2-0.4 to protect texture. Set denoiser \"none\" if the source is "
        "clean and the proposal over-denoises.\n\n"
        "UPSCALER (pick by content):\n"
        "- Photographic faces/people/general photos: \"Real-ESRGAN\" (robust, "
        "handles mild noise) or \"HAT-S\" for maximum fidelity on clean shots.\n"
        "- Fine high-frequency detail (hair, foliage, fabric; high "
        "detail_level/edge_density): \"HAT-S\".\n"
        "- Dense complex textures: \"DAT\". Very large images (>8 MP): "
        "\"Real-ESRGAN\" for speed/memory. Keep the proposed scale unless the "
        "source is tiny (then 4x-8x) or already large (then 2x).\n\n"
        "TONE / COLOR / BRIGHTNESS:\n"
        "- brightness<60: brightness +0.2..+0.6; brightness>200: -0.2..-0.5.\n"
        "- contrast<35 or dynamic_range<180: auto_tone 0.2-0.3 (use "
        "auto_contrast ~0.3 for very flat images).\n"
        "- color_temperature far from 1.0 (warm/cool cast): auto_color 0.2-0.6.\n"
        "- not grayscale and saturation_mean<40: prefer vibrance +0.2..+0.6 "
        "(protects skin tones, boosts muted colors more than saturated ones); "
        "use saturation only for a deliberate stylized/punchy look. Do not "
        "boost already-vivid images.\n\n"
        "SHARPNESS (avoid over-sharpening sharp images):\n"
        "- refocus recovers softness: blur_score<100 -> 0.6-0.8; 100-300 -> "
        "0.4-0.6; >300 -> 0.2-0.4. If edge_density>0.15 the image is already "
        "crisp - cut refocus/sharpen by half to prevent halos.\n"
        "- sharpen is a light final unsharp pass, 0.15-0.30.\n\n"
        + _NEW_ADJUSTERS_GUIDE + "\n"
        "PROCESSING ORDER (\"order\" = the exact sequence individual operations "
        "run in; each token is ONE method, applied to the output of the "
        "previous one, so sequence changes the result). Tokens (same names as "
        "the parameter keys above):\n  " + str(order_steps) + "\n"
        "- Recommended default: " + str(order_steps) + ".\n"
        "- Why this order: denoise FIRST so later steps do not amplify noise; "
        + ("deblur on the native-resolution image before upscaling; " if allow_deblur
           else "") +
        "white-balance/tone (auto_color, auto_tone, auto_contrast) before "
        "brightness/contrast/saturation; do all tonal/color work before "
        "upscale so the upscaler sees corrected pixels; upscale in the middle; "
        "refocus then sharpen LAST so final detail is crisp without amplifying "
        "earlier artifacts.\n"
        "- Only the operations you actually use need to appear; list each one "
        "you set a parameter for, in the order you want it applied. Deviate "
        "with a reason, e.g.: heavy noise -> also denoise right before "
        "\"upscale\"; tiny source -> move \"upscale\" earlier so adjusters and "
        "\"sharpen\" act on more pixels; always keep \"sharpen\" after "
        "\"upscale\".\n"
        "- Omit \"order\" entirely to accept the recommended default.\n\n"
    )
    if allow_deblur:
        guide += _DEBLUR_GUIDE + "\n"
        deblur_rule = ""
    else:
        deblur_rule = (
            "Deblur/SmartDeblur is DISABLED for this request - never include a "
            "\"deblur\" key in your output.\n"
        )
    if allow_icedit:
        guide += _ICEDIT_GUIDE + "\n"
        icedit_rule = ""
    else:
        icedit_rule = (
            "Content-editing via image diffusion is DISABLED for this request "
            "- never include a semantic edit key in your output.\n"
        )
    if allow_face:
        guide += _FACE_GUIDE + "\n"
        face_rule = ""
    else:
        face_rule = (
            "Face restoration is DISABLED for this request - never include a "
            "\"face\" key in your output.\n"
        )
    blend_note = (_BLEND_NOTE + "\n") if blend_enabled else ""

    return (
        _style_block(style_directive) +
        "You are an expert photo-restoration retoucher. You are shown an image "
        "and its measured properties. Your goal is a flawless, natural result: "
        "sharp where it should be, clean of noise, correctly exposed and "
        "color-balanced, with no visible artifacts.\n\n"
        "Image metrics (blur_score: lower = blurrier; values are measured, not "
        "targets):\n" + summary + "\n\n"
        "Algorithmically proposed config (JSON):\n"
        + json.dumps(base_config, ensure_ascii=False) + "\n\n"
        + guide +
        "OUTPUT - return ONLY a JSON object with any of these optional keys to "
        "override the proposal. OMIT every key you would leave unchanged. Use "
        "the exact value ranges shown:\n"
        + schema + deblur_rule + icedit_rule + face_rule + blend_note +
        "Respond with the JSON object only - no prose, no code fences, no "
        "explanation."
    )


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response (robust to fences)."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end + 1] if start != -1 and end > start else ""
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _clamp(value, lo, hi):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return None


def validate_overrides(raw: dict, allow_deblur: bool = True,
                       allow_icedit: bool = True,
                       allow_face: bool = True) -> dict:
    """Keep only known keys with valid values; clamp numeric ranges.

    When *allow_deblur* is False any ``deblur`` block is dropped, so automatic
    operations never enable SmartDeblur while its checkbox is off.
    When *allow_icedit* is False any ``icedit`` block is dropped similarly.
    When *allow_face* is False any ``face`` block is dropped similarly.
    """
    out: dict = {}
    if not isinstance(raw, dict):
        return out

    if "blend" in raw:
        # The advisor never controls blend (Task 6A owns that surface via the
        # control-panel checkbox); only known keys below are copied into
        # *out*, so an unsolicited "blend" key from the model is dropped here
        # like any other unrecognized top-level key.
        log.debug("LLMAdvisor: ignoring unsolicited 'blend' key from model response")

    if raw.get("scale") in _VALID_SCALES:
        out["scale"] = int(raw["scale"])
    if isinstance(raw.get("enhance_only"), bool):
        out["enhance_only"] = raw["enhance_only"]
    if raw.get("upscaler") in _KNOWN_UPSCALERS:
        out["upscaler"] = raw["upscaler"]
    if raw.get("denoiser") in _KNOWN_DENOISERS or raw.get("denoiser") == "none":
        out["denoiser"] = raw["denoiser"]

    for key, lo, hi in [
        ("denoise_strength", 0.0, 1.0),
        ("auto_tone", 0.0, 1.0),
        ("auto_contrast", 0.0, 1.0),
        ("auto_color", 0.0, 1.0),
        ("sharpness", 0.0, 1.0),
        ("sharpen", 0.0, 1.0),
        ("refocus", 0.0, 1.0),
        ("saturation", -1.0, 1.0),
        ("brightness", -1.0, 1.0),
        ("contrast", -1.0, 1.0),
        ("auto_levels", 0.0, 1.0),
        ("clarity", 0.0, 1.0),
        ("dehaze", 0.0, 1.0),
        ("white_balance", 0.0, 1.0),
        ("dodge_burn", 0.0, 1.0),
        ("skin_smooth", 0.0, 1.0),
        ("shadows_lift", 0.0, 1.0),
        ("highlights_tame", 0.0, 1.0),
        ("vibrance", -1.0, 1.0),
    ]:
        if key in raw:
            v = _clamp(raw[key], lo, hi)
            if v is not None:
                out[key] = round(v, 3)

    if allow_deblur:
        deblur = _validate_deblur(raw.get("deblur"))
        if deblur:
            out["deblur"] = deblur

    if allow_icedit:
        icedit = _validate_icedit(raw.get("icedit"))
        if icedit:
            out["icedit"] = icedit

    if allow_face:
        face = _validate_face(raw.get("face"))
        if face:
            out["face"] = face

    optics = _validate_optics(raw.get("optics"))
    if optics:
        out["optics"] = optics

    split_tone = _validate_split_tone(raw.get("split_tone"))
    if split_tone:
        out["split_tone"] = split_tone

    order = _validate_order(raw.get("order"), allow_deblur, allow_icedit, allow_face)
    if order:
        out["order"] = order

    return out


def _validate_order(value, allow_deblur: bool = True,
                    allow_icedit: bool = True,
                    allow_face: bool = True) -> list:
    """Keep only known step tokens, in the given sequence, de-duplicated.

    When *allow_deblur* is False the ``deblur`` step is dropped so SmartDeblur
    cannot sneak back in via the order list. When *allow_icedit* is False the
    ``icedit`` step is similarly dropped. When *allow_face* is False the
    ``face_restore`` step is similarly dropped.
    """
    if not isinstance(value, list):
        return []
    allowed = list(_PIPELINE_STEPS)
    if not allow_deblur:
        allowed = [s for s in allowed if s != "deblur"]
    if not allow_icedit:
        allowed = [s for s in allowed if s != "icedit"]
    if not allow_face:
        allowed = [s for s in allowed if s != "face_restore"]
    seq = []
    for token in value:
        if token in allowed and token not in seq:
            seq.append(token)
    return seq


def _validate_deblur(d) -> dict:
    """Validate the full SmartDeblur parameter block; clamp to known ranges."""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    if isinstance(d.get("enabled"), bool):
        out["enabled"] = d["enabled"]
    if isinstance(d.get("auto"), bool):
        out["auto"] = d["auto"]
    if isinstance(d.get("edge_taper"), bool):
        out["edge_taper"] = d["edge_taper"]
    if d.get("blur_type") in _KNOWN_BLUR_TYPES:
        out["blur_type"] = d["blur_type"]
    if d.get("method") in _KNOWN_DEBLUR_METHODS:
        out["method"] = d["method"]
    for key, (lo, hi) in _DEBLUR_RANGES.items():
        if key in d:
            v = _clamp(d[key], lo, hi)
            if v is not None:
                out[key] = int(v) if key == "tv_iterations" else round(v, 2)
    return out


def _validate_icedit(d) -> dict:
    """Validate an ICEdit block: non-empty clamped instruction + known variant."""
    if not isinstance(d, dict):
        return {}
    instruction = d.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        return {}
    out = {
        "enabled": bool(d.get("enabled", True)),
        "instruction": instruction.strip()[:_ICEDIT_MAX_INSTRUCTION],
        "variant": d["variant"] if d.get("variant") in _KNOWN_ICEDIT_VARIANTS
        else "normal",
    }
    return out


def _validate_face(d) -> dict:
    """Validate a CodeFormer face block; clamp fidelity to [0,1]."""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    if isinstance(d.get("enabled"), bool):
        out["enabled"] = d["enabled"]
    if "fidelity" in d:
        v = _clamp(d["fidelity"], 0.0, 1.0)
        if v is not None:
            out["fidelity"] = round(v, 2)
    if isinstance(d.get("upscale_background"), bool):
        out["upscale_background"] = d["upscale_background"]
    return out


def _validate_optics(d) -> dict:
    """Validate an Optics block; clamp vignette/ca. Non-numeric fields dropped."""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    if "vignette" in d:
        v = _clamp(d["vignette"], -1.0, 1.0)
        if v is not None:
            out["vignette"] = round(v, 3)
    if "ca" in d:
        v = _clamp(d["ca"], 0.0, 1.0)
        if v is not None:
            out["ca"] = round(v, 3)
    return out


def _validate_split_tone(d) -> dict:
    """Validate a Split Toning block; hues wrap via %360, rest clamped."""
    if not isinstance(d, dict):
        return {}
    out: dict = {}
    for key in ("shadow_hue", "highlight_hue"):
        if key in d:
            try:
                out[key] = round(float(d[key]) % 360.0, 2)
            except (TypeError, ValueError):
                pass
    if "saturation" in d:
        v = _clamp(d["saturation"], 0.0, 1.0)
        if v is not None:
            out["saturation"] = round(v, 3)
    if "balance" in d:
        v = _clamp(d["balance"], -1.0, 1.0)
        if v is not None:
            out["balance"] = round(v, 3)
    return out


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """Merge validated overrides onto a copy of the algorithmic base config."""
    cfg = json.loads(json.dumps(base_config))  # deep copy of JSON-safe dict
    cfg.setdefault("denoise", {})
    cfg.setdefault("adjust", {})
    cfg.setdefault("upscale", {})
    cfg.setdefault("post", {})

    if "scale" in overrides:
        cfg["scale"] = overrides["scale"]
        if isinstance(cfg.get("upscale"), dict) and cfg["upscale"].get("plugin"):
            cfg["upscale"]["scale"] = overrides["scale"]
    if "enhance_only" in overrides:
        cfg["enhance_only"] = overrides["enhance_only"]
    if "upscaler" in overrides:
        scale = overrides.get("scale", cfg.get("scale", 4))
        cfg["upscale"] = {"plugin": overrides["upscaler"], "scale": scale}

    # explicit denoiser selection takes priority over strength-only tuning
    if "denoiser" in overrides:
        if overrides["denoiser"] == "none":
            cfg["denoise"] = {}
        else:
            strength = overrides.get("denoise_strength", 0.3)
            cfg["denoise"] = {overrides["denoiser"]: {"strength": strength}}
    elif "denoise_strength" in overrides:
        strength = overrides["denoise_strength"]
        if strength <= 0:
            cfg["denoise"] = {}
        elif cfg["denoise"]:
            name = next(iter(cfg["denoise"]))
            if isinstance(cfg["denoise"][name], dict):
                cfg["denoise"][name]["strength"] = strength
            else:
                cfg["denoise"][name] = {"strength": strength}
        else:
            cfg["denoise"]["NL-Means"] = {"strength": strength}

    # auto adjusters: presence with strength>0 enables, 0 removes
    for key, plugin in [("auto_tone", "Auto Tone"),
                        ("auto_contrast", "Auto Contrast"),
                        ("auto_color", "Auto Color"),
                        ("sharpness", "Sharpness"),
                        ("auto_levels", "Auto Levels"),
                        ("clarity", "Clarity"),
                        ("dehaze", "Dehaze"),
                        ("white_balance", "White Balance"),
                        ("dodge_burn", "Dodge & Burn"),
                        ("skin_smooth", "Skin Smooth")]:
        if key in overrides:
            v = overrides[key]
            if v <= 1e-3:
                cfg["adjust"].pop(plugin, None)
            else:
                cfg["adjust"][plugin] = {"strength": v}

    # shadows_lift/highlights_tame: two override keys merge into one plugin
    # config (ShadowsHighlightsPlugin takes both params together). A partial
    # override (only one of the two keys present) must not zero out the
    # other side's algorithmic value - merge onto the existing base config.
    if "shadows_lift" in overrides or "highlights_tame" in overrides:
        cur = cfg["adjust"].get("Shadows/Highlights") or {}
        shadows = overrides.get("shadows_lift", cur.get("shadows", 0.0))
        highlights = overrides.get("highlights_tame", cur.get("highlights", 0.0))
        if shadows <= 1e-3 and highlights <= 1e-3:
            cfg["adjust"].pop("Shadows/Highlights", None)
        else:
            cfg["adjust"]["Shadows/Highlights"] = {
                "shadows": shadows, "highlights": highlights,
                "radius": cur.get("radius", 30),
            }

    # vibrance: signed like saturation/brightness/contrast, but the plugin
    # takes the sign directly in "strength" - no separate "direction" key.
    if "vibrance" in overrides:
        v = overrides["vibrance"]
        if abs(v) < 1e-3:
            cfg["adjust"].pop("Vibrance", None)
        else:
            cfg["adjust"]["Vibrance"] = {"strength": v}

    # optics/split_tone: already-validated dicts merge straight into "adjust";
    # empty/absent removes the corresponding plugin config.
    if "optics" in overrides:
        o = overrides["optics"]
        if o:
            cfg["adjust"]["Optics"] = o
        else:
            cfg["adjust"].pop("Optics", None)
    if "split_tone" in overrides:
        st = overrides["split_tone"]
        if st:
            cfg["adjust"]["Split Toning"] = st
        else:
            cfg["adjust"].pop("Split Toning", None)

    # signed adjusters: direction from sign
    for key, plugin in [("saturation", "Saturation"),
                        ("brightness", "Brightness"),
                        ("contrast", "Contrast")]:
        if key in overrides:
            amount = overrides[key]
            if abs(amount) < 1e-3:
                cfg["adjust"].pop(plugin, None)
            else:
                cfg["adjust"][plugin] = {
                    "strength": abs(amount),
                    "direction": "up" if amount >= 0 else "down",
                }

    if "sharpen" in overrides:
        cfg["post"]["sharpen"] = overrides["sharpen"]
    if "refocus" in overrides:
        if overrides["refocus"] <= 1e-3:
            cfg["post"].pop("refocus", None)
        else:
            cfg["post"]["refocus"] = {"strength": overrides["refocus"]}

    if "deblur" in overrides:
        d = overrides["deblur"]
        if d.get("enabled") is False:
            cfg["deblur"] = {}
        else:
            cur = cfg.get("deblur") if isinstance(cfg.get("deblur"), dict) else {}
            merged = dict(cur)
            shape_keys = ("blur_type", "radius", "angle", "smooth",
                          "edge_feather", "correction_strength")
            for k, v in d.items():
                if k != "enabled":
                    merged[k] = v
            # explicit shape parameters mean manual mode unless model said auto
            if any(k in d for k in shape_keys) and "auto" not in d:
                merged["auto"] = False
            elif "auto" not in merged:
                merged["auto"] = True
            cfg["deblur"] = merged

    if "icedit" in overrides:
        ic = overrides["icedit"]
        if ic.get("enabled") is False:
            cfg["icedit"] = {}
        else:
            cur = cfg.get("icedit") if isinstance(cfg.get("icedit"), dict) else {}
            merged = dict(cur)
            for k, v in ic.items():
                merged[k] = v
            cfg["icedit"] = merged

    if "face" in overrides:
        fc = overrides["face"]
        if fc.get("enabled") is False:
            cfg["face"] = {}
        else:
            cur = cfg.get("face") if isinstance(cfg.get("face"), dict) else {}
            merged = dict(cur)
            for k, v in fc.items():
                merged[k] = v
            merged.setdefault("enabled", True)
            cfg["face"] = merged

    if "order" in overrides:
        cfg["order"] = list(overrides["order"])

    return cfg


class LLMAdvisor:
    """Refines auto-config via a local GGUF model, with graceful fallback."""

    def __init__(self, models_dir: Optional[Path] = None,
                 model_filename: Optional[str] = None,
                 n_gpu_layers: int = -1):
        if models_dir is None:
            # bundled local models live at upscaler/models/models/; resolve it
            # directly to avoid importing the torch-heavy models.manager module.
            models_dir = Path(__file__).resolve().parent.parent / "models" / "models"
        self.models_dir = Path(models_dir)
        self._model_filename = model_filename
        # How many model layers to offload to the GPU. -1 = offload all when the
        # llama-cpp build supports CUDA (the "use GPU when possible" default); a
        # CPU-only build silently ignores this and runs on the CPU. Pass 0 to
        # force CPU. Applies to both the text and vision (CLIP/mmproj) paths.
        self.n_gpu_layers = n_gpu_layers
        self._llm = None  # lazily constructed llama_cpp.Llama
        self._is_vision = False

    @staticmethod
    def runtime_available() -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except Exception:
            return False

    def model_path(self) -> Optional[Path]:
        if self._model_filename:
            p = self.models_dir / self._model_filename
            return p if p.exists() else None
        for name in _PREFERRED_MODELS:
            p = self.models_dir / name
            if p.exists():
                return p
        # any gguf as a last resort
        ggufs = sorted(self.models_dir.glob("*.gguf"))
        return ggufs[0] if ggufs else None

    def available(self) -> bool:
        return self.runtime_available() and self.model_path() is not None

    def _model_family(self, model_path: Optional[Path] = None) -> str:
        name = (model_path or self.model_path() or Path("")).name.lower()
        if "gemma" in name:
            return "gemma"
        if "qwen" in name:
            return "qwen"
        return "other"

    def _mmproj_path(self, model_path: Optional[Path] = None) -> Optional[Path]:
        """Find the multimodal projector matching the selected model's family."""
        candidates = sorted(self.models_dir.glob("*mmproj*.gguf")) + \
            sorted(self.models_dir.glob("*mmproj*.bin"))
        if not candidates:
            return None
        family = self._model_family(model_path)
        if family != "other":
            for c in candidates:
                if family in c.name.lower():
                    return c
        return candidates[0]

    def _vision_handler(self, mmproj_path: Path):
        """Pick the chat handler matching the model family; raise if none fits."""
        from llama_cpp import llama_chat_format as fmt
        family = self._model_family()
        # Prefer the family-specific handler, falling back to Llava (the most
        # widely supported). Handler availability depends on the installed
        # llama-cpp-python version.
        # Handler class names vary across llama-cpp-python versions (e.g. 0.3.31
        # renamed Gemma3->Gemma4 and dropped Qwen2VL). List newest first, then
        # older aliases, then the generic Llava fallback; getattr skips names
        # absent in the installed build, so this stays version-robust.
        order = {
            "gemma": ["Gemma4ChatHandler", "Gemma3ChatHandler",
                      "Llava15ChatHandler"],
            "qwen": ["Qwen25VLChatHandler", "Qwen2VLChatHandler",
                     "Llava15ChatHandler"],
            "other": ["Llava15ChatHandler"],
        }[family]
        for cls_name in order:
            cls = getattr(fmt, cls_name, None)
            if cls is not None:
                return cls(clip_model_path=str(mmproj_path))
        raise RuntimeError("no compatible vision chat handler available")

    def _build_llm(self):
        """Construct the Llama model, preferring vision; fall back to text."""
        import llama_cpp
        model_path = str(self.model_path())
        mmproj = self._mmproj_path()
        if mmproj is not None:
            try:
                handler = self._vision_handler(mmproj)
                llm = llama_cpp.Llama(model_path=model_path, n_ctx=4096,
                                      n_gpu_layers=self.n_gpu_layers,
                                      chat_handler=handler, verbose=False)
                self._is_vision = True
                return llm
            except Exception as exc:
                log.warning("Vision init failed (%s); using text-only LLM", exc)
        llm = llama_cpp.Llama(model_path=model_path, n_ctx=4096,
                              n_gpu_layers=self.n_gpu_layers, verbose=False)
        self._is_vision = False
        return llm

    @staticmethod
    def _encode_image_data_uri(image: np.ndarray) -> str:
        """Encode an RGB image as a base64 PNG data URI for multimodal input."""
        import base64
        import cv2
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("failed to encode preview image")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    def _generate(self, prompt: str, image: Optional[np.ndarray]) -> str:
        """Run the model. Best-effort; raises on any backend failure."""
        if self._llm is None:
            self._llm = self._build_llm()

        if getattr(self, "_is_vision", False) and image is not None:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": self._encode_image_data_uri(image)}},
            ]
        else:
            content = prompt

        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            temperature=0.2, max_tokens=512,
        )
        return out["choices"][0]["message"]["content"]

    def refine(self, image: Optional[np.ndarray], analysis: dict,
               base_config: dict,
               generate: Optional[Callable[[str, Optional[np.ndarray]], str]] = None,
               allow_deblur: bool = True,
               allow_icedit: bool = True,
               allow_face: bool = True,
               blend_enabled: bool = False,
               style_directive: str = "",
               ) -> dict:
        """Return a refined config, or *base_config* unchanged on any problem.

        When *allow_deblur* is False the model is never asked for - and is never
        allowed to return - SmartDeblur parameters. When *allow_icedit* is False
        ICEdit instruction parameters are similarly excluded. When *allow_face*
        is False face restoration parameters are similarly excluded.
        *blend_enabled* only informs the model that a later blend step may run
        (see ``_BLEND_NOTE``); the advisor never controls blend itself.
        """
        gen = generate
        if gen is None:
            if not self.available():
                log.info("LLMAdvisor unavailable; using algorithmic config")
                return base_config
            gen = self._generate

        try:
            summary = build_analysis_summary(analysis)
            prompt = build_prompt(summary, base_config,
                                  allow_deblur=allow_deblur,
                                  allow_icedit=allow_icedit,
                                  allow_face=allow_face,
                                  blend_enabled=blend_enabled,
                                  style_directive=style_directive)
            response = gen(prompt, image)
            raw = extract_json(response)
            overrides = validate_overrides(raw, allow_deblur=allow_deblur,
                                           allow_icedit=allow_icedit,
                                           allow_face=allow_face)
            if not overrides:
                log.info("LLMAdvisor returned no usable overrides")
                return base_config
            refined = apply_overrides(base_config, overrides)
            log.info("LLMAdvisor applied overrides: %s", overrides)
            return refined
        except Exception as exc:
            log.warning("LLMAdvisor failed (%s); using algorithmic config", exc)
            return base_config

    def evaluate(self, image, analysis: dict,
                 allow_deblur: bool = True, allow_icedit: bool = True,
                 allow_face: bool = True,
                 generate=None, style_directive: str = "") -> dict:
        """Judge a processed result. Returns {"satisfied", "config"}.

        config is None when satisfied (or on any failure); otherwise an
        enhance-only refinement config (never upscales).
        """
        gen = generate
        if gen is None:
            if not self.available():
                return {"satisfied": True, "config": None}
            gen = self._generate
        try:
            summary = build_analysis_summary(analysis)
            prompt = build_evaluation_prompt(
                summary, allow_deblur=allow_deblur, allow_icedit=allow_icedit,
                allow_face=allow_face, style_directive=style_directive)
            response = gen(prompt, image)
            raw = extract_json(response)
            satisfied, overrides = validate_evaluation(
                raw, allow_deblur=allow_deblur, allow_icedit=allow_icedit,
                allow_face=allow_face)
            if satisfied or not overrides:
                return {"satisfied": True, "config": None}
            base = {"enhance_only": True, "denoise": {}, "adjust": {},
                    "upscale": {}, "post": {}}
            cfg = apply_overrides(base, overrides)
            cfg["enhance_only"] = True
            log.info("LLMAdvisor refinement overrides: %s", overrides)
            return {"satisfied": False, "config": cfg}
        except Exception as exc:
            log.warning("LLMAdvisor.evaluate failed (%s); stopping loop", exc)
            return {"satisfied": True, "config": None}
