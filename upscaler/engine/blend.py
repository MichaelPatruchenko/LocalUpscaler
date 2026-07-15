"""Режимы наложения Photoshop на numpy (float32, RGB 0..1).

Формулы разделяемых режимов и HSL-примитивов (lum/set_lum/sat/set_sat/
clip_color) — по спецификации Adobe (PDF blend modes). Вход uint8
конвертируется в float и обратно; альфа-канал базы сохраняется как есть.
"""
import numpy as np

BLEND_MODES = (
    "normal", "dissolve",
    "darken", "multiply", "color_burn", "linear_burn",
    "lighten", "screen", "color_dodge", "linear_dodge",
    "overlay", "soft_light", "hard_light", "vivid_light",
    "linear_light", "pin_light", "hard_mix",
    "difference", "exclusion", "subtract", "divide",
    "hue", "saturation", "color", "luminosity",
)

# Режим -> i18n-ключ (slug) его подписи (blend.<mode> в upscaler/ui/i18n.py).
# Используйте blend_mode_label(mode) для получения переведённого текста.
BLEND_MODE_LABELS = {
    "normal": "blend.normal", "dissolve": "blend.dissolve",
    "darken": "blend.darken", "multiply": "blend.multiply",
    "color_burn": "blend.color_burn", "linear_burn": "blend.linear_burn",
    "lighten": "blend.lighten", "screen": "blend.screen",
    "color_dodge": "blend.color_dodge", "linear_dodge": "blend.linear_dodge",
    "overlay": "blend.overlay", "soft_light": "blend.soft_light",
    "hard_light": "blend.hard_light", "vivid_light": "blend.vivid_light",
    "linear_light": "blend.linear_light", "pin_light": "blend.pin_light",
    "hard_mix": "blend.hard_mix",
    "difference": "blend.difference", "exclusion": "blend.exclusion",
    "subtract": "blend.subtract", "divide": "blend.divide",
    "hue": "blend.hue", "saturation": "blend.saturation",
    "color": "blend.color", "luminosity": "blend.luminosity",
}


def blend_mode_label(mode: str) -> str:
    """Переведённая подпись режима наложения для текущего языка UI."""
    from upscaler.ui.i18n import tr
    key = BLEND_MODE_LABELS.get(mode, mode)
    return tr(key)

_EPS = 1e-12


# --- HSL-примитивы (Adobe) --------------------------------------------------

def _lum(c):
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _clip_color(c):
    l = _lum(c)[..., None]
    mn = c.min(axis=-1, keepdims=True)
    mx = c.max(axis=-1, keepdims=True)
    c = np.where(mn < 0.0, l + (c - l) * l / np.maximum(l - mn, _EPS), c)
    c = np.where(mx > 1.0, l + (c - l) * (1.0 - l) /
                 np.maximum(mx - l, _EPS), c)
    return np.clip(c, 0.0, 1.0)


def _set_lum(c, l_target):
    return _clip_color(c + (l_target - _lum(c))[..., None])


def _sat(c):
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c, s_target):
    mn = c.min(axis=-1, keepdims=True)
    mx = c.max(axis=-1, keepdims=True)
    rng = np.maximum(mx - mn, _EPS)
    scaled = (c - mn) / rng * s_target[..., None]
    return np.where((mx - mn) > _EPS, scaled, 0.0)


# --- Разделяемые режимы -----------------------------------------------------

def _soft_light(b, s):
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(b))
    return np.where(s <= 0.5,
                    b - (1.0 - 2.0 * s) * b * (1.0 - b),
                    b + (2.0 * s - 1.0) * (d - b))


def _color_burn(b, s):
    return 1.0 - np.minimum(1.0, (1.0 - b) / np.maximum(s, _EPS))


def _color_dodge(b, s):
    return np.minimum(1.0, b / np.maximum(1.0 - s, _EPS))


def _overlay(b, s):
    return np.where(b <= 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))


def _vivid_light(b, s):
    return np.where(s <= 0.5, _color_burn(b, 2.0 * s),
                    _color_dodge(b, 2.0 * (s - 0.5)))


_SEPARABLE = {
    "normal": lambda b, s: s,
    "darken": np.minimum,
    "multiply": lambda b, s: b * s,
    "color_burn": _color_burn,
    "linear_burn": lambda b, s: np.clip(b + s - 1.0, 0.0, 1.0),
    "lighten": np.maximum,
    "screen": lambda b, s: 1.0 - (1.0 - b) * (1.0 - s),
    "color_dodge": _color_dodge,
    "linear_dodge": lambda b, s: np.clip(b + s, 0.0, 1.0),
    "overlay": _overlay,
    "soft_light": _soft_light,
    "hard_light": lambda b, s: _overlay(s, b),
    "vivid_light": _vivid_light,
    "linear_light": lambda b, s: np.clip(b + 2.0 * s - 1.0, 0.0, 1.0),
    "pin_light": lambda b, s: np.where(s <= 0.5,
                                       np.minimum(b, 2.0 * s),
                                       np.maximum(b, 2.0 * s - 1.0)),
    "hard_mix": lambda b, s: np.where(b + s >= 1.0, 1.0, 0.0),
    "difference": lambda b, s: np.abs(b - s),
    "exclusion": lambda b, s: b + s - 2.0 * b * s,
    "subtract": lambda b, s: np.clip(b - s, 0.0, 1.0),
    "divide": lambda b, s: np.clip(b / np.maximum(s, _EPS), 0.0, 1.0),
}

_COMPONENT = {
    "hue": lambda b, s: _set_lum(_set_sat(s, _sat(b)), _lum(b)),
    "saturation": lambda b, s: _set_lum(_set_sat(b, _sat(s)), _lum(b)),
    "color": lambda b, s: _set_lum(s, _lum(b)),
    "luminosity": lambda b, s: _set_lum(b, _lum(s)),
}


def _to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, True
    return img.astype(np.float32), False


def blend(base: np.ndarray, layer: np.ndarray, mode: str,
          opacity: float = 1.0, seed: int = 0) -> np.ndarray:
    """Наложить *layer* на *base* режимом *mode* с прозрачностью *opacity*.

    Формы должны совпадать по HxW; альфа базы (4-й канал) сохраняется.
    Возвращает массив в dtype базы. dissolve детерминирован через *seed*.
    """
    if mode not in BLEND_MODES:
        raise ValueError(f"Неизвестный режим наложения: {mode!r}")
    if base.shape[:2] != layer.shape[:2]:
        raise ValueError(
            f"Формы не совпадают: {base.shape[:2]} vs {layer.shape[:2]}")
    opacity = float(np.clip(opacity, 0.0, 1.0))

    bf, was_uint8 = _to_float(base)
    sf, _ = _to_float(layer)
    alpha = bf[..., 3:4] if bf.ndim == 3 and bf.shape[2] == 4 else None
    b = np.clip(bf[..., :3], 0.0, 1.0)
    s = np.clip(sf[..., :3], 0.0, 1.0)

    if mode == "dissolve":
        rng = np.random.default_rng(seed)
        mask = (rng.random(b.shape[:2]) < opacity)[..., None]
        out = np.where(mask, s, b)
    else:
        blended = (_SEPARABLE[mode](b, s) if mode in _SEPARABLE
                   else _COMPONENT[mode](b, s))
        blended = np.clip(blended, 0.0, 1.0)
        out = b * (1.0 - opacity) + blended * opacity

    if alpha is not None:
        out = np.concatenate([out, alpha], axis=-1)
    if was_uint8:
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out.astype(np.float32)
