"""Чистая геометрия ручных зон лиц на холсте.

Зоны — нормализованные [x, y, w, h] или [x, y, w, h, angle] (доли 0..1
изображения; angle в градусах, поворот вокруг центра зоны). Маппинг
widget<->image повторяет BeforeAfterCanvas.paintEvent: изображение
масштабируется zoom'ом и центрируется с учётом пана.
"""
import math

MIN_ZONE_NORM = 0.01  # минимальная сторона зоны (доля изображения)


def unpack_zone(rect):
    """(x, y, w, h, angle); терпим к 4- и 5-элементным зонам."""
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    angle = float(rect[4]) if len(rect) > 4 else 0.0
    return float(x), float(y), float(w), float(h), angle


def _rotate(px, py, cx, cy, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    dx, dy = px - cx, py - cy
    return cx + dx * c - dy * s, cy + dx * s + dy * c


def zone_corners(rect):
    """4 угла зоны (норм.), повёрнутые вокруг центра на angle."""
    x, y, w, h, angle = unpack_zone(rect)
    cx, cy = x + w / 2.0, y + h / 2.0
    base = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if angle == 0.0:
        return base
    return [_rotate(px, py, cx, cy, angle) for px, py in base]


def rotation_handle_norm(rect, offset=0.06):
    """Норм. точка маркера поворота над серединой верхней грани."""
    x, y, w, h, angle = unpack_zone(rect)
    cx = x + w / 2.0
    hx, hy = cx, y - offset
    return _rotate(hx, hy, cx, y + h / 2.0, angle) if angle else (hx, hy)


def angle_from_handle_drag(rect, nx, ny):
    """Угол зоны по позиции курсора относительно центра (0 = вверх)."""
    x, y, w, h, _ = unpack_zone(rect)
    cx, cy = x + w / 2.0, y + h / 2.0
    deg = math.degrees(math.atan2(nx - cx, -(ny - cy)))
    return deg % 360.0


def image_rect_on_widget(widget_w, widget_h, img_w, img_h,
                         zoom, pan_x, pan_y):
    """(x, y, w, h) нарисованного изображения в координатах виджета."""
    scaled_w = img_w * zoom
    scaled_h = img_h * zoom
    x = (widget_w - scaled_w) / 2.0 + pan_x
    y = (widget_h - scaled_h) / 2.0 + pan_y
    return x, y, scaled_w, scaled_h


def widget_to_norm(px, py, widget_w, widget_h, img_w, img_h,
                   zoom, pan_x, pan_y):
    """Точка виджета -> нормализованные координаты изображения (кламп 0..1)."""
    ix, iy, iw, ih = image_rect_on_widget(widget_w, widget_h, img_w, img_h,
                                          zoom, pan_x, pan_y)
    if iw <= 0 or ih <= 0:
        return 0.0, 0.0
    nx = (px - ix) / iw
    ny = (py - iy) / ih
    return min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)


def norm_to_widget(nx, ny, widget_w, widget_h, img_w, img_h,
                   zoom, pan_x, pan_y):
    """Нормализованные координаты -> точка виджета."""
    ix, iy, iw, ih = image_rect_on_widget(widget_w, widget_h, img_w, img_h,
                                          zoom, pan_x, pan_y)
    return ix + nx * iw, iy + ny * ih


def rect_from_points(x0, y0, x1, y1):
    """Нормализованный [x, y, w, h] по двум углам; кламп и мин. размер."""
    x, y = min(x0, x1), min(y0, y1)
    w, h = abs(x1 - x0), abs(y1 - y0)
    x = min(max(x, 0.0), 1.0 - MIN_ZONE_NORM)
    y = min(max(y, 0.0), 1.0 - MIN_ZONE_NORM)
    w = min(max(w, MIN_ZONE_NORM), 1.0 - x)
    h = min(max(h, MIN_ZONE_NORM), 1.0 - y)
    # Round away float noise (e.g. abs(0.2 - 0.6) == 0.39999999999999997)
    # so equality comparisons on normalized coords behave as expected.
    return [round(x, 9), round(y, 9), round(w, 9), round(h, 9)]


def hit_test_zone(zones, nx, ny):
    """Индекс верхней (последней в списке) зоны, содержащей точку, или None."""
    for i in range(len(zones) - 1, -1, -1):
        x, y, w, h, angle = unpack_zone(zones[i])
        cx, cy = x + w / 2.0, y + h / 2.0
        lx, ly = (_rotate(nx, ny, cx, cy, -angle) if angle else (nx, ny))
        if x <= lx <= x + w and y <= ly <= y + h:
            return i
    return None


def hit_test_handle(zones, px, py, widget_w, widget_h, img_w, img_h,
                    zoom, pan_x, pan_y, handle_px=8):
    """(индекс зоны, угол 0=TL,1=TR,2=BR,3=BL), если точка на маркере угла."""
    for i in range(len(zones) - 1, -1, -1):
        for ci, (cnx, cny) in enumerate(zone_corners(zones[i])):
            wx, wy = norm_to_widget(cnx, cny, widget_w, widget_h, img_w, img_h,
                                    zoom, pan_x, pan_y)
            if abs(px - wx) <= handle_px and abs(py - wy) <= handle_px:
                return i, ci
    return None


def hit_test_rotation_handle(zones, px, py, widget_w, widget_h, img_w, img_h,
                             zoom, pan_x, pan_y, handle_px=10):
    """Индекс зоны, если точка на маркере поворота, иначе None."""
    for i in range(len(zones) - 1, -1, -1):
        hnx, hny = rotation_handle_norm(zones[i])
        wx, wy = norm_to_widget(hnx, hny, widget_w, widget_h, img_w, img_h,
                                zoom, pan_x, pan_y)
        if abs(px - wx) <= handle_px and abs(py - wy) <= handle_px:
            return i
    return None


def move_zone(rect, dnx, dny):
    """Сдвиг зоны на (dnx, dny) с клампом в границы изображения."""
    x, y, w, h, angle = unpack_zone(rect)
    x = min(max(x + dnx, 0.0), 1.0 - w)
    y = min(max(y + dny, 0.0), 1.0 - h)
    return [x, y, w, h, angle]


def resize_zone(rect, corner, nx, ny):
    """Перетащить угол зоны в (nx, ny); противоположный угол фиксирован."""
    x, y, w, h, angle = unpack_zone(rect)
    anchors = {0: (x + w, y + h), 1: (x, y + h), 2: (x, y), 3: (x + w, y)}
    ax, ay = anchors[int(corner)]
    new = rect_from_points(ax, ay, min(max(nx, 0.0), 1.0),
                           min(max(ny, 0.0), 1.0))
    return new + [angle]
