"""Генерация скриншотов приложения для README (docs/screenshots/).

Запуск: py -3.10 scripts/make_screenshots.py
Использует QWidget.grab() (без реального показа окна), поэтому работает и
в headless-окружении с offscreen-платформой.
"""
import os
import sys
from pathlib import Path

# NB: we deliberately do NOT force QT_QPA_PLATFORM=offscreen. On Windows the
# offscreen platform has no font database, so every label renders as tofu
# boxes. The native platform loads system fonts (readable text); QWidget.grab()
# captures widgets without ever showing a window. Set QT_QPA_PLATFORM=offscreen
# in the environment yourself only if running truly headless (text will be
# unreadable there).

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def _demo_image():
    import numpy as np
    import cv2
    rng = np.random.default_rng(7)
    img = np.full((480, 640, 3), (90, 110, 130), np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, 560, 2)
        c = tuple(int(v) for v in rng.integers(40, 230, 3))
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 70, int(y) + 60), c, -1)
    cv2.circle(img, (320, 240), 90, (200, 180, 160), -1)
    return img


def _save(widget, name):
    widget.resize(widget.sizeHint())
    pix = widget.grab()
    path = OUT / f"{name}.png"
    pix.save(str(path))
    print("saved", path.name, pix.width(), "x", pix.height())


def _isolate_history():
    """Point history at an empty temp dir so MainWindow doesn't pop the modal
    "resume previous session?" dialog (which would block forever offscreen),
    and so the user's real ~/.upscaler/history is never touched or deleted."""
    import tempfile
    import upscaler.config as cfg
    tmp = Path(tempfile.mkdtemp(prefix="uexp-shots-"))
    cfg.HISTORY_DIR = tmp / "history"
    cfg.HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    _isolate_history()
    from upscaler.ui.main_window import MainWindow
    win = MainWindow()
    win.resize(1400, 900)
    win._current_image = _demo_image()
    win.canvas.set_before_image(win._current_image)
    win.canvas.set_after_image(win._current_image)

    # Главное окно (одиночный режим)
    win.single_mode_rb.setChecked(True)
    _save(win, "01-main-single")

    # Панель коррекции — развернуть секцию «Коррекция» и её подсекции
    cp = win.control_panel
    if "adjust" in cp._sections:
        cp._sections["adjust"].set_expanded(True)
        for k in ("adjust_tone", "adjust_color", "adjust_detail"):
            if k in cp._sections:
                cp._sections[k].set_expanded(True)
    _save(cp, "02-correction-sections")

    # Вкладки
    win.deblur_mode_rb.setChecked(True); _save(win.deblur_panel, "03-smartdeblur")
    win.icedit_mode_rb.setChecked(True); _save(win.icedit_panel, "04-icedit")
    win.blend_mode_rb.setChecked(True); _save(win.blend_panel, "05-blend")
    win.face_mode_rb.setChecked(True)
    win.face_panel.set_zones([[0.25, 0.2, 0.3, 0.35], [0.55, 0.45, 0.2, 0.25]])
    _save(win.face_panel, "06-face-zones")
    _save(win.history_panel, "07-history")

    try:
        win.controller.stop_engine()
    except Exception:
        pass
    print("done ->", OUT)


if __name__ == "__main__":
    main()
