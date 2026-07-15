"""Оценка «эффективного разрешения» изображения по радиальному спектру.

Определяет, содержит ли изображение реальную детализацию на уровне
номинального размера, или это апсемпл/«мыло». Возвращаемый фактор — до
какой доли размера изображение можно уменьшить без потери реальной
детализации (1.0 = не уменьшать).

Метрика считается на нескольких кропах ПОЛНОРАЗМЕРНОГО изображения (без
ресемплинга — он уничтожил бы спектральную сигнатуру апсемпла): доля
энергии спектра выше кандидатной частоты отсечки в средне-высокочастотной
полосе (>= 10% Найквиста). Шум распределён по всем частотам, поэтому
шумные изображения честно защищены от даунскейла (консервативно). Итог —
максимум по кропам: одного резкого региона достаточно, чтобы не уменьшать.
"""
import numpy as np

FACTORS = (0.75, 0.66, 0.5, 0.33, 0.25)
_EPS = 0.05        # доля энергии выше отсечки, при которой полоса «пуста»
_BASE_BAND = 0.10  # нормировочная полоса: r >= 10% Найквиста
_CROP = 512
_MIN_RMAX = 16


def _crop_factor(g: np.ndarray) -> float:
    """Фактор для одного квадратного кропа (float64, полный размер пикселей)."""
    side = g.shape[0]
    win = g * np.outer(np.hanning(side), np.hanning(side))
    power = np.abs(np.fft.fftshift(np.fft.fft2(win))) ** 2
    c = side // 2
    if c < _MIN_RMAX:
        return 1.0
    yy, xx = np.ogrid[:side, :side]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
    radial = np.bincount(r.ravel(), weights=power.ravel())[:c]
    base_lo = int(_BASE_BAND * c)
    band = radial[base_lo:]
    total = band.sum()
    if total <= 0:
        return 1.0
    best = 1.0
    for f in FACTORS:
        cutoff = int(round(f * c)) - base_lo
        above = band[cutoff:].sum() if 0 <= cutoff < len(band) else 0.0
        if above / total < _EPS:
            best = f
        else:
            break  # факторы убывают: дальше полоса только шире
    return best


def effective_downscale_factor(gray: np.ndarray, crop: int = _CROP) -> float:
    """Квантованный фактор даунскейла (из FACTORS) либо 1.0."""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = g[:, :, :3].mean(axis=2)
    g = g.astype(np.float64)
    if g.size and g.max() <= 1.5:  # float [0,1] -> [0,255]
        g = g * 255.0
    h, w = g.shape[:2]
    side = min(crop, h, w)
    if side < 2 * _MIN_RMAX:
        return 1.0
    cy, cx = (h - side) // 2, (w - side) // 2
    positions = {(0, 0), (0, w - side), (h - side, 0),
                 (h - side, w - side), (cy, cx)}
    return max(_crop_factor(g[y:y + side, x:x + side])
               for y, x in positions)
