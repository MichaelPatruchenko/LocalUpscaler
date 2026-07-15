import numpy as np
import pytest
from upscaler.plugins.icedit.incontext import (
    resize_to_width, build_diptych, build_instruction_prompt, extract_right_half,
)


def _img(h=40, w=60):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


def test_resize_to_width_keeps_aspect_and_even_height():
    out = resize_to_width(_img(40, 60), 512)
    assert out.shape[1] == 512
    assert out.shape[0] % 2 == 0
    assert out.dtype == np.uint8


def test_build_diptych_shapes_and_halves():
    img = _img(32, 48)
    canvas, mask = build_diptych(img)
    h, w = img.shape[:2]
    assert canvas.shape == (h, 2 * w, 3)
    assert mask.shape == (h, 2 * w)
    # left half equals the source image
    assert np.array_equal(canvas[:, :w], img)
    # mask: left half zero, right half 255
    assert mask[:, :w].max() == 0
    assert mask[:, w:].min() == 255


def test_extract_right_half_round_trip():
    img = _img(20, 30)
    canvas, _ = build_diptych(img)
    right = extract_right_half(canvas)
    assert right.shape == img.shape
    assert np.array_equal(right, img)  # right half initialized to the image copy


def test_build_instruction_prompt_contains_diptych_template_and_instruction():
    p = build_instruction_prompt("make the hair green")
    assert "diptych" in p.lower()
    assert "on the right" in p.lower()
    assert "make the hair green" in p
