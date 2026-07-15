"""Side-by-side comparison dialog with synchronized zoom/pan."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QWidget, QPushButton, QSplitter,
)
from PySide6.QtCore import Qt, QPoint, QRectF, QTimer
from PySide6.QtGui import QImage, QPainter, QWheelEvent, QMouseEvent
import numpy as np

from upscaler.ui.i18n import tr


class CompareCanvas(QWidget):
    """Single image canvas with zoom and pan, linkable to a partner."""

    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._label = label_text
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._panning = False
        self._last_mouse_pos = QPoint()
        self._partner: "CompareCanvas | None" = None
        self.setMinimumSize(300, 200)

    def set_image(self, image: np.ndarray):
        if image.dtype != np.uint8:
            if image.dtype in (np.float32, np.float64):
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            elif image.dtype == np.uint16:
                image = (image / 257).astype(np.uint8)
        if image.ndim == 2:
            h, w = image.shape
            self._image = QImage(image.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        else:
            h, w, c = image.shape
            bpl = w * c
            fmt = QImage.Format.Format_RGB888 if c == 3 else QImage.Format.Format_RGBA8888
            self._image = QImage(image.tobytes(), w, h, bpl, fmt)
        self.update()

    def set_label_text(self, text: str):
        self._label = text
        self.update()

    def set_zoom(self, zoom: float):
        self._zoom = max(0.1, min(zoom, 20.0))
        self.update()

    def set_pan(self, offset: QPoint):
        self._pan_offset = QPoint(offset)
        self.update()

    def fit_to_view(self):
        if self._image:
            wr = self.width() / self._image.width()
            hr = self.height() / self._image.height()
            self._zoom = min(wr, hr)
            self._pan_offset = QPoint(0, 0)
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if self._image:
            sw = int(self._image.width() * self._zoom)
            sh = int(self._image.height() * self._zoom)
            x = (self.width() - sw) // 2 + self._pan_offset.x()
            y = (self.height() - sh) // 2 + self._pan_offset.y()
            painter.drawImage(QRectF(x, y, sw, sh), self._image)
        painter.setPen(Qt.GlobalColor.white)
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(10, 25, self._label)
        painter.end()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._zoom = max(0.1, min(self._zoom * factor, 20.0))
        if self._partner:
            self._partner.set_zoom(self._zoom)
            self._partner.set_pan(self._pan_offset)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.position().toPoint() - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = event.position().toPoint()
            if self._partner:
                self._partner.set_pan(QPoint(self._pan_offset))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._panning = False


class CompareDialog(QDialog):
    """Side-by-side comparison of original and processed image."""

    def __init__(self, original: np.ndarray, processed: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("compare.window_title"))
        self.setMinimumSize(900, 500)
        self.resize(1200, 700)

        layout = QVBoxLayout(self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_canvas = CompareCanvas(tr("compare.original_label"))
        self.right_canvas = CompareCanvas(tr("compare.result_label"))
        self.left_canvas._partner = self.right_canvas
        self.right_canvas._partner = self.left_canvas

        splitter.addWidget(self.left_canvas)
        splitter.addWidget(self.right_canvas)
        splitter.setSizes([600, 600])
        layout.addWidget(splitter)

        self.close_btn = QPushButton(tr("compare.close_btn"))
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        self.left_canvas.set_image(original)
        self.right_canvas.set_image(processed)
        QTimer.singleShot(100, self._fit_both)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты диалога из текущего языка."""
        self.setWindowTitle(tr("compare.window_title"))
        self.left_canvas.set_label_text(tr("compare.original_label"))
        self.right_canvas.set_label_text(tr("compare.result_label"))
        self.close_btn.setText(tr("compare.close_btn"))

    def _fit_both(self):
        self.left_canvas.fit_to_view()
        self.right_canvas.fit_to_view()
