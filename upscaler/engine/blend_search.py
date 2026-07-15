"""Жадный автоподбор blend-рецепта по кандидатам конвейера.

Поиск идёт на прокси (<=512px), финальная композиция — apply_recipe в
полном разрешении. Детерминирован: кандидаты обходятся в
отсортированном порядке ключей, dissolve в поиске не участвует.
"""
import logging

import cv2
import numpy as np

from upscaler.engine.blend import blend
from upscaler.engine.quality_score import quality_score

log = logging.getLogger(__name__)

# Режимы, которые пробуем, когда кандидат сильнее базы в данном свойстве.
PROPERTY_MODES = {
    "sharpness": ("luminosity", "overlay", "soft_light"),
    "contrast": ("overlay", "soft_light"),
    "colorfulness": ("color", "saturation"),
    "exposure": ("screen", "soft_light"),
    "noise_penalty": ("normal",),
}
OPACITIES = (0.25, 0.5, 0.75)


def make_proxy(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Уменьшенная копия для поиска (только downscale, INTER_AREA)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (max(int(w * scale), 8), max(int(h * scale), 8)),
                      interpolation=cv2.INTER_AREA)


def _candidate_modes(cand_parts: dict, base_parts: dict) -> list:
    modes = []
    for prop, prop_modes in PROPERTY_MODES.items():
        if cand_parts.get(prop, 0.0) > base_parts.get(prop, 0.0):
            for m in prop_modes:
                if m not in modes:
                    modes.append(m)
    if "normal" not in modes:
        modes.append("normal")
    return modes


def greedy_blend_search(candidates: dict, max_layers: int = 3,
                        eps: float = 0.005) -> dict:
    """Подобрать рецепт наложений, максимизирующий quality_score.

    *candidates* — прокси одинакового размера {имя: изображение}.
    """
    if not candidates:
        return {"base": None, "layers": [], "score": 0.0}
    parts = {k: quality_score(v) for k, v in sorted(candidates.items())}
    base_key = max(sorted(parts), key=lambda k: parts[k]["score"])
    composite = candidates[base_key].copy()
    comp_parts = parts[base_key]
    used = {base_key}
    layers = []

    for _ in range(max_layers):
        best = None  # (gain, key, mode, opacity, new_img, new_parts)
        for key in sorted(candidates):
            if key in used:
                continue
            layer = candidates[key]
            if layer.shape[:2] != composite.shape[:2]:
                continue
            for mode in _candidate_modes(parts[key], comp_parts):
                for opacity in OPACITIES:
                    trial = blend(composite, layer, mode, opacity)
                    trial_parts = quality_score(trial)
                    gain = trial_parts["score"] - comp_parts["score"]
                    if best is None or gain > best[0]:
                        best = (gain, key, mode, opacity, trial, trial_parts)
        if best is None or best[0] <= eps:
            break
        gain, key, mode, opacity, composite, comp_parts = best
        used.add(key)
        layers.append({"source": key, "mode": mode, "opacity": opacity})
        log.info("Blend search: +%s via %s@%.0f%% (gain %.4f)",
                 key, mode, opacity * 100, gain)

    return {"base": base_key, "layers": layers,
            "score": comp_parts["score"]}


def apply_recipe(recipe: dict, images: dict) -> np.ndarray:
    """Полноразмерная композиция рецепта. Источники ресайзятся к базе."""
    base = images[recipe["base"]].copy()
    h, w = base.shape[:2]
    for layer in recipe.get("layers", []):
        src = images.get(layer["source"])
        if src is None:
            continue
        if src.shape[:2] != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LANCZOS4)
        base = blend(base, src, layer["mode"], layer["opacity"])
    return base
