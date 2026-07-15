import numpy as np
from upscaler.ui.canvas_widget import compute_canvas_images

A = np.zeros((4, 4, 3), dtype=np.uint8)
B = np.ones((4, 4, 3), dtype=np.uint8)


def test_neither_returns_none():
    assert compute_canvas_images(None, None) == (None, None)


def test_primary_only_is_display_mode():
    before, after = compute_canvas_images(A, None)
    assert before is A and after is A


def test_both_is_compare_mode():
    before, after = compute_canvas_images(A, B)
    assert before is A and after is B


def test_secondary_only_falls_back_to_display():
    before, after = compute_canvas_images(None, B)
    assert before is B and after is B
