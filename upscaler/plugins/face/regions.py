"""Хелперы ручных зон лиц: денормализация, запас, синтетические лендмарки.

Чистые numpy-функции без моделей — юнит-тестируются без загрузок. Зоны
хранятся нормализованными [x, y, w, h] (доли 0..1 от изображения), поэтому
инвариантны к масштабированию конвейера.
"""
import numpy as np

from upscaler.plugins.face.align import FFHQ_512_TEMPLATE

_MIN_SIDE_PX = 8  # вырожденные зоны меньше этого отбрасываются вызывающим


def denormalize_rect(rect, img_w: int, img_h: int) -> tuple:
    """Нормализованный [x, y, w, h] -> пиксельный (x, y, w, h), кламп в кадр."""
    nx, ny, nw, nh = (float(rect[i]) for i in range(4))
    x0 = min(max(nx, 0.0), 1.0) * img_w
    y0 = min(max(ny, 0.0), 1.0) * img_h
    x1 = min(max(nx + nw, 0.0), 1.0) * img_w
    y1 = min(max(ny + nh, 0.0), 1.0) * img_h
    x = min(int(round(x0)), img_w - 1)
    y = min(int(round(y0)), img_h - 1)
    w = max(int(round(x1 - x0)), 1)
    h = max(int(round(y1 - y0)), 1)
    w = max(min(w, img_w - x), 1)
    h = max(min(h, img_h - y), 1)
    return x, y, w, h


def expand_rect(x: int, y: int, w: int, h: int, img_w: int, img_h: int,
                margin: float = 0.2) -> tuple:
    """Расширить прямоугольник на *margin* с каждой стороны, кламп в кадр."""
    dx = int(round(w * margin))
    dy = int(round(h * margin))
    ex = max(x - dx, 0)
    ey = max(y - dy, 0)
    ew = min(x + w + dx, img_w) - ex
    eh = min(y + h + dy, img_h) - ey
    return ex, ey, ew, eh


def synthetic_landmarks(x: float, y: float, w: float, h: float,
                        angle: float = 0.0) -> np.ndarray:
    """5 точек FFHQ, вписанные в зону и повёрнутые на *angle* вокруг центра.

    align_face по таким лендмаркам отображает (в т.ч. наклонённую) зону на
    канву 512x512 — реставрация выравнивает лицо корректно.
    """
    scale = np.array([w / 512.0, h / 512.0], dtype=np.float32)
    offset = np.array([x, y], dtype=np.float32)
    pts = (FFHQ_512_TEMPLATE * scale + offset).astype(np.float32)
    if angle:
        cx, cy = x + w / 2.0, y + h / 2.0
        r = np.radians(angle)
        c, s = np.cos(r), np.sin(r)
        d = pts - np.array([cx, cy], np.float32)
        pts = np.stack([
            cx + d[:, 0] * c - d[:, 1] * s,
            cy + d[:, 0] * s + d[:, 1] * c,
        ], axis=1).astype(np.float32)
    return pts
