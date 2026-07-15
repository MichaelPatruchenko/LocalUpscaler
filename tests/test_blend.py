"""Режимы наложения Photoshop: формулы против эталонных значений."""
import numpy as np
import pytest
from upscaler.engine.blend import BLEND_MODES, BLEND_MODE_LABELS, blend, blend_mode_label


def _px(*rgb):
    """1x1x3 float32 изображение."""
    return np.array(rgb, dtype=np.float32).reshape(1, 1, 3)


def _val(mode, b, s, opacity=1.0):
    """Скалярное наложение серых пикселей (все каналы одинаковы)."""
    out = blend(_px(b, b, b), _px(s, s, s), mode, opacity)
    return float(out[0, 0, 0])


def test_mode_registry_complete():
    assert len(BLEND_MODES) == 25
    assert len(set(BLEND_MODES)) == 25
    for m in BLEND_MODES:
        assert m in BLEND_MODE_LABELS and blend_mode_label(m).strip()


def test_reference_values_separable_modes():
    cases = {
        ("normal", 0.3, 0.8): 0.8,
        ("darken", 0.3, 0.8): 0.3,
        ("multiply", 0.5, 0.5): 0.25,
        ("color_burn", 0.5, 0.5): 0.0,
        ("linear_burn", 0.3, 0.5): 0.0,      # 0.3+0.5-1 -> clip
        ("lighten", 0.3, 0.8): 0.8,
        ("screen", 0.5, 0.5): 0.75,
        ("color_dodge", 0.5, 0.5): 1.0,
        ("linear_dodge", 0.7, 0.5): 1.0,      # clip
        ("overlay", 0.25, 0.5): 0.25,         # s=0.5 нейтрален
        ("overlay", 0.75, 0.5): 0.75,
        ("hard_light", 0.5, 0.25): 0.25,      # overlay со сменой аргументов
        ("vivid_light", 0.5, 0.25): 0.0,
        ("vivid_light", 0.5, 0.75): 1.0,
        ("linear_light", 0.4, 0.5): 0.4,      # b+2s-1
        ("pin_light", 0.3, 0.2): 0.3,
        ("pin_light", 0.3, 0.8): 0.6,
        ("hard_mix", 0.6, 0.5): 1.0,
        ("hard_mix", 0.4, 0.5): 0.0,
        ("difference", 0.7, 0.2): 0.5,
        ("exclusion", 0.5, 0.5): 0.5,          # b+s-2bs
        ("subtract", 0.7, 0.2): 0.5,
        ("divide", 0.25, 0.5): 0.5,
    }
    for (mode, b, s), expected in cases.items():
        assert _val(mode, b, s) == pytest.approx(expected, abs=1e-6), mode


def test_soft_light_neutral_at_half():
    for b in (0.0, 0.2, 0.5, 0.9, 1.0):
        assert _val("soft_light", b, 0.5) == pytest.approx(b, abs=1e-6)


def test_luminosity_transfers_gray_level():
    base = _px(0.2, 0.4, 0.6)
    layer = _px(0.5, 0.5, 0.5)
    out = blend(base, layer, "luminosity")
    # lum(base)=0.3*0.2+0.59*0.4+0.11*0.6=0.362; сдвиг каналов на 0.138
    assert np.allclose(out, _px(0.338, 0.538, 0.738), atol=1e-3)


def test_hue_on_gray_base_stays_gray():
    base = _px(0.5, 0.5, 0.5)          # насыщенность базы = 0
    layer = _px(1.0, 0.0, 0.0)         # красный
    out = blend(base, layer, "hue")
    assert np.allclose(out[0, 0, 0], out[0, 0, 1], atol=1e-6)
    assert np.allclose(out[0, 0, 1], out[0, 0, 2], atol=1e-6)


def test_color_mode_takes_layer_chroma_base_lum():
    base = _px(0.362, 0.362, 0.362)
    layer = _px(1.0, 0.0, 0.0)
    out = blend(base, layer, "color")
    # Красный с яркостью базы: R > G == B
    assert out[0, 0, 0] > out[0, 0, 1]
    assert out[0, 0, 1] == pytest.approx(out[0, 0, 2], abs=1e-6)


def test_opacity_lerp():
    assert _val("normal", 0.0, 1.0, opacity=0.25) == pytest.approx(0.25, abs=1e-6)
    assert _val("multiply", 0.5, 0.5, opacity=0.5) == pytest.approx(
        0.5 * 0.5 + 0.25 * 0.5, abs=1e-6)


def test_dissolve_deterministic_and_density():
    base = np.zeros((64, 64, 3), np.float32)
    layer = np.ones((64, 64, 3), np.float32)
    a = blend(base, layer, "dissolve", opacity=0.3, seed=7)
    b = blend(base, layer, "dissolve", opacity=0.3, seed=7)
    assert np.array_equal(a, b)
    frac = float(a[:, :, 0].mean())
    assert 0.2 < frac < 0.4                      # плотность ~ opacity
    assert set(np.unique(a)) <= {0.0, 1.0}       # без полутонов


def test_uint8_roundtrip_and_dtype():
    base = np.full((4, 4, 3), 128, np.uint8)
    layer = np.full((4, 4, 3), 64, np.uint8)
    out = blend(base, layer, "multiply")
    assert out.dtype == np.uint8
    assert abs(int(out[0, 0, 0]) - round(128 * 64 / 255)) <= 1


def test_unknown_mode_and_shape_mismatch_raise():
    img = np.zeros((4, 4, 3), np.float32)
    with pytest.raises(ValueError):
        blend(img, img, "bogus")
    with pytest.raises(ValueError):
        blend(img, np.zeros((2, 2, 3), np.float32), "normal")
