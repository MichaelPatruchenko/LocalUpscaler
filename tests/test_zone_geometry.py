"""Чистая геометрия зон: маппинг, hit-тесты, построение/ресайз."""
from upscaler.ui.zone_geometry import (
    image_rect_on_widget, widget_to_norm, norm_to_widget, rect_from_points,
    hit_test_zone, hit_test_handle, move_zone, resize_zone,
)

# Изображение 100x100, виджет 200x200, zoom 1.0, без пана:
# картинка занимает (50, 50, 100, 100) на виджете.
P = dict(widget_w=200, widget_h=200, img_w=100, img_h=100,
         zoom=1.0, pan_x=0, pan_y=0)


def test_image_rect_centered():
    assert image_rect_on_widget(**P) == (50.0, 50.0, 100.0, 100.0)


def test_widget_to_norm_roundtrip():
    nx, ny = widget_to_norm(100, 100, **P)
    assert (nx, ny) == (0.5, 0.5)
    px, py = norm_to_widget(nx, ny, **P)
    assert (px, py) == (100.0, 100.0)


def test_widget_to_norm_clamps_outside():
    nx, ny = widget_to_norm(0, 0, **P)
    assert (nx, ny) == (0.0, 0.0)
    nx, ny = widget_to_norm(500, 500, **P)
    assert (nx, ny) == (1.0, 1.0)


def test_widget_to_norm_with_zoom_and_pan():
    q = dict(P, zoom=2.0, pan_x=10, pan_y=-10)
    # image rect: (200-200)/2+10=10, (200-200)/2-10=-10, 200, 200
    assert image_rect_on_widget(**q) == (10.0, -10.0, 200.0, 200.0)
    nx, ny = widget_to_norm(110, 90, **q)
    assert (round(nx, 3), round(ny, 3)) == (0.5, 0.5)


def test_rect_from_points_any_corner_order():
    assert rect_from_points(0.6, 0.7, 0.2, 0.3) == [0.2, 0.3, 0.4, 0.4]


def test_rect_from_points_min_size():
    r = rect_from_points(0.5, 0.5, 0.5, 0.5)
    assert r[2] >= 0.01 and r[3] >= 0.01


def test_hit_test_zone_topmost():
    zones = [[0.1, 0.1, 0.5, 0.5], [0.3, 0.3, 0.5, 0.5]]
    assert hit_test_zone(zones, 0.4, 0.4) == 1  # верхняя (последняя)
    assert hit_test_zone(zones, 0.15, 0.15) == 0
    assert hit_test_zone(zones, 0.95, 0.95) is None


def test_hit_test_handle_corners():
    zones = [[0.25, 0.25, 0.5, 0.5]]  # на виджете: (75,75)-(125,125)
    assert hit_test_handle(zones, 75, 75, **P) == (0, 0)    # TL
    assert hit_test_handle(zones, 125, 75, **P) == (0, 1)   # TR
    assert hit_test_handle(zones, 125, 125, **P) == (0, 2)  # BR
    assert hit_test_handle(zones, 75, 125, **P) == (0, 3)   # BL
    assert hit_test_handle(zones, 100, 100, **P) is None


def test_move_zone_clamped():
    assert move_zone([0.8, 0.8, 0.15, 0.15], 0.5, 0.5) == [0.85, 0.85, 0.15, 0.15, 0.0]


def test_resize_zone_br_corner():
    r = resize_zone([0.2, 0.2, 0.2, 0.2], 2, 0.6, 0.7)
    assert [round(v, 3) for v in r] == [0.2, 0.2, 0.4, 0.5, 0.0]


def test_resize_zone_flip_safe():
    # Утащили BR-угол левее/выше TL — прямоугольник не вырождается
    r = resize_zone([0.4, 0.4, 0.2, 0.2], 2, 0.1, 0.1)
    assert r[2] >= 0.01 and r[3] >= 0.01


# --- Блок F: повёрнутые зоны -------------------------------------------------
import math
from upscaler.ui.zone_geometry import (
    unpack_zone, zone_corners, rotation_handle_norm,
    angle_from_handle_drag, hit_test_rotation_handle,
)

Q = dict(widget_w=200, widget_h=200, img_w=100, img_h=100,
         zoom=1.0, pan_x=0, pan_y=0)  # image at (50,50,100,100)


def test_unpack_zone_tolerates_4_and_5():
    assert unpack_zone([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4, 0.0)
    assert unpack_zone([0.1, 0.2, 0.3, 0.4, 30.0]) == (0.1, 0.2, 0.3, 0.4, 30.0)


def test_zone_corners_axis_aligned():
    c = zone_corners([0.2, 0.2, 0.4, 0.4])   # angle 0
    assert [tuple(round(v, 6) for v in p) for p in c] == [
        (0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]


def test_zone_corners_rotated_90_keeps_center():
    rect = [0.3, 0.35, 0.4, 0.3, 90.0]
    c = zone_corners(rect)
    cx = sum(p[0] for p in c) / 4
    cy = sum(p[1] for p in c) / 4
    assert abs(cx - 0.5) < 1e-6 and abs(cy - 0.5) < 1e-6  # центр не сместился


def test_rotation_handle_above_top_when_unrotated():
    hx, hy = rotation_handle_norm([0.25, 0.25, 0.5, 0.5])
    assert abs(hx - 0.5) < 1e-6      # над серединой верхней грани
    assert hy < 0.25                  # выше зоны


def test_angle_from_handle_drag_right_is_90ish():
    # Курсор справа от центра -> ~90° (или -90 в зависимости от знака);
    # проверяем, что угол монотонно меняется и в диапазоне.
    rect = [0.25, 0.25, 0.5, 0.5]
    a_up = angle_from_handle_drag(rect, 0.5, 0.0)      # прямо вверх
    a_right = angle_from_handle_drag(rect, 1.0, 0.5)   # вправо
    assert abs(((a_up) % 360)) < 1e-6 or abs((a_up % 360) - 360) < 1e-6
    assert 80 <= (a_right % 360) <= 100 or 260 <= (a_right % 360) <= 280


def test_hit_test_rotation_handle():
    zones = [[0.25, 0.25, 0.5, 0.5]]
    hx, hy = rotation_handle_norm(zones[0])
    from upscaler.ui.zone_geometry import norm_to_widget
    wx, wy = norm_to_widget(hx, hy, **Q)
    assert hit_test_rotation_handle(zones, wx, wy, **Q) == 0
    assert hit_test_rotation_handle(zones, wx + 40, wy + 40, **Q) is None


def test_hit_test_zone_rotated():
    # Зона повёрнута на 45°; точка в центре всегда внутри
    from upscaler.ui.zone_geometry import hit_test_zone
    zones = [[0.3, 0.3, 0.4, 0.4, 45.0]]
    assert hit_test_zone(zones, 0.5, 0.5) == 0


def test_move_and_resize_preserve_angle():
    from upscaler.ui.zone_geometry import move_zone, resize_zone
    assert move_zone([0.2, 0.2, 0.3, 0.3, 25.0], 0.05, 0.05)[4] == 25.0
    assert resize_zone([0.2, 0.2, 0.3, 0.3, 25.0], 2, 0.6, 0.7)[4] == 25.0
