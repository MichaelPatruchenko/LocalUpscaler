"""Галерея выбора варианта обработки (сетка 2×2, зум/панорамирование).

Открывается на каждый round_ready оркестратора «4 вариантов». Возвращает
индекс выбранного кандидата (`selected_index`); None — галерея закрыта без
выбора (сессия завершается, изображение не меняется).
"""
from PySide6.QtWidgets import (
    QDialog, QGridLayout, QVBoxLayout, QLabel, QPushButton, QWidget,
)
from PySide6.QtCore import Qt, QTimer

from upscaler.ui.compare_dialog import CompareCanvas
from upscaler.ui.i18n import tr


class FourVariantsGalleryDialog(QDialog):
    """Модальный выбор из кандидатов раунда. Ячейки failed/satisfied видимы,
    но недоступны (в раундах 2+ кандидатов может быть меньше четырёх)."""

    def __init__(self, candidates: list[dict], iteration: int = 1,
                 max_iter: int = 1, parent=None):
        super().__init__(parent)
        self.selected_index: int | None = None
        self._canvases: list[CompareCanvas] = []

        self.setWindowTitle(tr("variants.gallery_title"))
        self.setMinimumSize(900, 700)
        self.resize(1200, 850)

        layout = QVBoxLayout(self)
        header = QLabel(tr("variants.gallery_round_title",
                           iter=iteration, max=max_iter))
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        for i, cand in enumerate(candidates):
            grid.addWidget(self._build_cell(i, cand), i // 2, i % 2)
        layout.addWidget(grid_host, stretch=1)

        QTimer.singleShot(100, self._fit_all)

    def _build_cell(self, index: int, cand: dict) -> QWidget:
        cell = QWidget()
        v = QVBoxLayout(cell)
        v.setContentsMargins(4, 4, 4, 4)

        name = tr(cand.get("name_key", ""))
        status = cand.get("status")
        canvas = CompareCanvas(name)
        if status == "done" and cand.get("result_image") is not None:
            canvas.set_image(cand["result_image"])
            self._canvases.append(canvas)
        v.addWidget(canvas, stretch=1)

        if status == "done":
            metrics = cand.get("metrics") or {}
            sub = tr("variants.metrics_label",
                     brisque=float(metrics.get("brisque", 0.0)),
                     niqe=float(metrics.get("niqe", 0.0)))
        elif status == "satisfied":
            sub = tr("variants.satisfied_label")
        else:
            sub = tr("variants.failed_label")
        sub_label = QLabel(sub)
        sub_label.setStyleSheet("color: gray; font-size: 11px;")
        sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub_label)

        btn = QPushButton(tr("variants.choose_btn"))
        btn.setEnabled(status == "done")
        btn.clicked.connect(lambda _=False, i=index: self._on_choose(i))
        v.addWidget(btn)
        return cell

    def _on_choose(self, index: int) -> None:
        self.selected_index = index
        self.accept()

    def _fit_all(self) -> None:
        for canvas in self._canvases:
            canvas.fit_to_view()
