"""Тесты чистых хелперов ручных зон лиц."""
import numpy as np
from upscaler.plugins.face.regions import (
    denormalize_rect, expand_rect, synthetic_landmarks,
)
from upscaler.plugins.face.align import FFHQ_512_TEMPLATE, align_face


def test_denormalize_basic():
    assert denormalize_rect([0.25, 0.5, 0.5, 0.25], 400, 200) == (100, 100, 200, 50)


def test_denormalize_clamps_to_image():
    x, y, w, h = denormalize_rect([0.9, 0.9, 0.5, 0.5], 100, 100)
    assert x + w <= 100 and y + h <= 100
    assert x >= 0 and y >= 0 and w > 0 and h > 0


def test_denormalize_negative_clamped():
    x, y, w, h = denormalize_rect([-0.2, -0.2, 0.5, 0.5], 100, 100)
    assert x == 0 and y == 0
    assert w > 0 and h > 0


def test_expand_rect_adds_margin():
    x, y, w, h = expand_rect(100, 100, 100, 100, 1000, 1000, margin=0.2)
    assert x == 80 and y == 80 and w == 140 and h == 140


def test_expand_rect_clamped_at_border():
    x, y, w, h = expand_rect(0, 0, 100, 100, 110, 110, margin=0.2)
    assert x == 0 and y == 0
    assert x + w <= 110 and y + h <= 110


def test_synthetic_landmarks_shape_and_position():
    lms = synthetic_landmarks(100, 200, 512, 512)
    assert lms.shape == (5, 2) and lms.dtype == np.float32
    # При w=h=512 шаблон просто сдвигается
    assert np.allclose(lms, FFHQ_512_TEMPLATE + np.array([100, 200]))


def test_synthetic_landmarks_scale():
    lms = synthetic_landmarks(0, 0, 256, 256)
    assert np.allclose(lms, FFHQ_512_TEMPLATE * 0.5, atol=1e-4)


def test_synthetic_landmarks_align_maps_rect_to_canvas():
    # align_face по синтетическим лендмаркам должен отобразить зону в канву
    # 512x512 почти тождественно (сдвиг+масштаб).
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[50:178, 100:228] = 255  # зона 128x128 в (100, 50)
    lms = synthetic_landmarks(100, 50, 128, 128)
    crop, m = align_face(img, lms)
    assert crop.shape[:2] == (512, 512)
    # Центр зоны должен стать центром канвы (аффинное отображение зоны на канву)
    cx, cy = 100 + 64, 50 + 64
    mapped = m @ np.array([cx, cy, 1.0])
    assert abs(mapped[0] - 256) < 2 and abs(mapped[1] - 256) < 2


def test_denormalize_far_edge_stays_inside():
    x, y, w, h = denormalize_rect([1.0, 0.0, 0.0, 0.5], 1000, 1000)
    assert x + w <= 1000 and y + h <= 1000 and w >= 1 and h >= 1
    x, y, w, h = denormalize_rect([0.0, 1.0, 0.5, 0.0], 1000, 1000)
    assert x + w <= 1000 and y + h <= 1000 and w >= 1 and h >= 1


def test_synthetic_landmarks_angle_rotates_about_center():
    up = synthetic_landmarks(100, 100, 200, 200, angle=0.0)
    rot = synthetic_landmarks(100, 100, 200, 200, angle=90.0)
    # центр (200,200) неподвижен; лендмарки повернулись
    assert not np.allclose(up, rot)
    cx, cy = 200.0, 200.0
    # средняя точка облака ~ центр в обоих случаях (шаблон симметричен по X)
    assert abs(up[:, 0].mean() - rot[:, 0].mean()) < 40


def test_synthetic_landmarks_angle0_matches_legacy():
    a = synthetic_landmarks(50, 60, 128, 128)
    b = synthetic_landmarks(50, 60, 128, 128, angle=0.0)
    assert np.allclose(a, b)
