"""Общий VersionListPanel: автомат выбора первичный/вторичный."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from upscaler.ui.version_list_panel import VersionListPanel

_app = QApplication.instance() or QApplication([])


def _panel():
    p = VersionListPanel()
    for v in (1, 2, 3):
        p.add_item(v, np.zeros((4, 4, 3), np.uint8), f"v{v}")
    return p


def test_primary_secondary_selection():
    p = _panel()
    got = []
    p.selection_changed.connect(lambda a, b: got.append((a, b)))
    p.list_widget.primary_clicked.emit(2)
    assert p.selected() == (2, None)
    p.list_widget.secondary_clicked.emit(3)
    assert p.selected() == (2, 3)


def test_lone_secondary_normalizes_to_primary():
    p = _panel()
    p.list_widget.secondary_clicked.emit(3)
    # одинокий вторичный трактуется как первичный (для показа/работы)
    assert p.selected() == (3, None)


def test_deselect_primary_leaves_secondary():
    p = _panel()
    p.list_widget.primary_clicked.emit(1)
    p.list_widget.secondary_clicked.emit(2)
    p.list_widget.primary_clicked.emit(1)  # снять первичный
    assert p.selected() == (2, None)


def test_clear_resets_selection():
    p = _panel()
    p.list_widget.primary_clicked.emit(1)
    p.clear()
    assert p.selected() == (None, None)
    assert p.list_widget.count() == 0


def test_add_item_makes_it_primary():
    p = VersionListPanel()
    p.add_item(5, np.zeros((4, 4, 3), np.uint8), "v5")
    assert p.selected() == (5, None)
