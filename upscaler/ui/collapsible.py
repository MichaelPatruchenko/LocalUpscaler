"""Сворачиваемая секция: заголовок-кнопка со стрелкой + контейнер."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.header_btn = QToolButton()
        self.header_btn.setText(title)
        self.header_btn.setCheckable(True)
        self.header_btn.setChecked(expanded)
        self.header_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header_btn.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; }")
        self.header_btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.header_btn.toggled.connect(self._on_toggled)

        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(8, 0, 0, 4)
        self._body.setVisible(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)
        outer.addWidget(self.header_btn)
        outer.addWidget(self._body)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def is_expanded(self) -> bool:
        return self.header_btn.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header_btn.setChecked(bool(expanded))

    def _on_toggled(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self.header_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.toggled.emit(checked)
