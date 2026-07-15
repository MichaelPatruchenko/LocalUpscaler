"""CollapsibleSection: сворачиваемая секция панели."""
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QLabel
from upscaler.ui.collapsible import CollapsibleSection

_app = QApplication.instance() or QApplication([])


def test_default_expanded_shows_body():
    s = CollapsibleSection("Секция")
    s.add_widget(QLabel("x"))
    assert s.is_expanded() is True
    assert not s._body.isHidden()


def test_collapsed_by_default_hides_body():
    s = CollapsibleSection("Секция", expanded=False)
    s.add_widget(QLabel("x"))
    assert s.is_expanded() is False
    assert s._body.isHidden()


def test_toggle_via_header_button():
    s = CollapsibleSection("Секция")
    got = []
    s.toggled.connect(lambda v: got.append(v))
    s.header_btn.click()
    assert s.is_expanded() is False and s._body.isHidden()
    assert got == [False]
    s.header_btn.click()
    assert s.is_expanded() is True and not s._body.isHidden()
    assert got == [False, True]


def test_set_expanded_programmatic():
    s = CollapsibleSection("Секция")
    s.set_expanded(False)
    assert s.is_expanded() is False
    s.set_expanded(True)
    assert s.is_expanded() is True
