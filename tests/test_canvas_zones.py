"""Режим разметки зон на холсте (программные события)."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from upscaler.ui.canvas_widget import BeforeAfterCanvas

_app = QApplication.instance() or QApplication([])


def _canvas():
    c = BeforeAfterCanvas()
    # BeforeAfterCanvas.__init__ sets a 400x300 minimum size; without
    # relaxing it, resize(200, 200) below is clamped and the widget stays
    # 400x300, breaking every pixel coordinate used in this test module.
    c.setMinimumSize(0, 0)
    c.resize(200, 200)
    c.set_before_image(np.full((100, 100, 3), 100, dtype=np.uint8))
    c.set_zoom(1.0)
    c.set_zone_edit_mode(True)
    return c


def _mouse(c, etype, x, y, button=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(etype, QPointF(x, y), button, button,
                     Qt.KeyboardModifier.NoModifier)
    if etype == QEvent.Type.MouseButtonPress:
        c.mousePressEvent(ev)
    elif etype == QEvent.Type.MouseMove:
        c.mouseMoveEvent(ev)
    else:
        c.mouseReleaseEvent(ev)


def test_drag_creates_zone():
    c = _canvas()
    got = []
    c.zones_changed.connect(lambda z: got.append(z))
    _mouse(c, QEvent.Type.MouseButtonPress, 75, 75)
    _mouse(c, QEvent.Type.MouseMove, 125, 125)
    _mouse(c, QEvent.Type.MouseButtonRelease, 125, 125)
    zones = c.get_zones()
    assert len(zones) == 1
    assert [round(v, 2) for v in zones[0]] == [0.25, 0.25, 0.5, 0.5, 0.0]
    assert got and got[-1] == zones


def test_click_selects_and_delete_removes():
    c = _canvas()
    c.set_zones([[0.25, 0.25, 0.5, 0.5]])
    _mouse(c, QEvent.Type.MouseButtonPress, 100, 100)
    _mouse(c, QEvent.Type.MouseButtonRelease, 100, 100)
    assert c.selected_zone() == 0
    c.delete_selected_zone()
    assert c.get_zones() == []


def test_drag_inside_moves_zone():
    c = _canvas()
    c.set_zones([[0.25, 0.25, 0.5, 0.5]])
    _mouse(c, QEvent.Type.MouseButtonPress, 100, 100)
    _mouse(c, QEvent.Type.MouseMove, 110, 110)
    _mouse(c, QEvent.Type.MouseButtonRelease, 110, 110)
    z = c.get_zones()[0]
    assert [round(v, 2) for v in z] == [0.35, 0.35, 0.5, 0.5, 0.0]


def test_edit_mode_off_keeps_panning():
    c = _canvas()
    c.set_zone_edit_mode(False)
    _mouse(c, QEvent.Type.MouseButtonPress, 100, 100)
    _mouse(c, QEvent.Type.MouseMove, 110, 110)
    _mouse(c, QEvent.Type.MouseButtonRelease, 110, 110)
    assert c.get_zones() == []  # зоны не создаются вне режима разметки


def test_drag_rotation_handle_sets_angle():
    c = _canvas()  # существующий хелпер в файле
    c.set_zones([[0.25, 0.25, 0.5, 0.5]])
    c.set_zone_edit_mode(True)
    # выбрать зону кликом по центру
    _mouse(c, QEvent.Type.MouseButtonPress, 100, 100)
    _mouse(c, QEvent.Type.MouseButtonRelease, 100, 100)
    from upscaler.ui.zone_geometry import rotation_handle_norm, norm_to_widget
    params = c._zone_params()
    hx, hy = rotation_handle_norm(c.get_zones()[0])
    wx, wy = norm_to_widget(hx, hy, **params)
    _mouse(c, QEvent.Type.MouseButtonPress, wx, wy)
    _mouse(c, QEvent.Type.MouseMove, 160, 100)  # тащим вправо
    _mouse(c, QEvent.Type.MouseButtonRelease, 160, 100)
    zone = c.get_zones()[0]
    assert len(zone) == 5
    assert zone[4] % 360 != 0.0          # угол изменился


def test_new_zone_has_angle_field():
    c = _canvas()
    c.set_zone_edit_mode(True)
    _mouse(c, QEvent.Type.MouseButtonPress, 75, 75)
    _mouse(c, QEvent.Type.MouseMove, 125, 125)
    _mouse(c, QEvent.Type.MouseButtonRelease, 125, 125)
    z = c.get_zones()[0]
    assert len(z) == 5 and z[4] == 0.0
