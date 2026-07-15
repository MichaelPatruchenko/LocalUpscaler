"""Before/after split canvas with synchronized zoom and pan."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSlider, QPushButton
from PySide6.QtCore import Qt, QPoint, QPointF, Signal, QRectF
from PySide6.QtGui import (QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
                            QColor, QPen, QPolygonF)
import numpy as np

from upscaler.ui import zone_geometry as zg


def compute_canvas_images(primary, secondary):
    """Decide the (before, after) images for the canvas from a selection.

    primary/secondary are np.ndarray or None.
      - both set       → (primary, secondary)   compare mode
      - primary only   → (primary, primary)      display mode
      - secondary only → (secondary, secondary)  display mode (defensive)
      - neither        → (None, None)             empty canvas
    """
    if primary is None and secondary is None:
        return None, None
    if primary is not None and secondary is not None:
        return primary, secondary
    show = primary if primary is not None else secondary
    return show, show


class BeforeAfterCanvas(QWidget):
    """Split view canvas for comparing before and after images."""

    zones_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_image: QImage | None = None
        self._after_image: QImage | None = None
        self._split_pos = 0.5  # 0-1, position of the splitter
        self._zoom = 1.0
        self._pan_offset = QPoint(0, 0)
        self._dragging_splitter = False
        self._panning = False
        self._last_mouse_pos = QPoint()
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self._zone_edit = False
        self._zones: list = []          # нормализованные [x, y, w, h]
        self._sel_zone: int | None = None
        self._zone_drag = None          # None | "new" | "move" | "resize"
        self._zone_drag_corner = 0
        self._zone_anchor = (0.0, 0.0)  # старт drag'а (норм.)
        self._zone_orig = None          # rect на момент начала drag'а
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_before_image(self, image: np.ndarray | None):
        """Set the 'before' image (left side)."""
        if image is not None:
            self._before_image = self._ndarray_to_qimage(image)
        else:
            self._before_image = None
        self.update()

    def set_after_image(self, image: np.ndarray | None):
        """Set the 'after' image (right side)."""
        if image is not None:
            self._after_image = self._ndarray_to_qimage(image)
        else:
            self._after_image = None
        self.update()

    def set_zoom(self, zoom: float):
        self._zoom = max(0.1, min(zoom, 20.0))
        self.update()

    def fit_to_view(self):
        """Reset zoom and pan to fit image in widget."""
        img = self._after_image or self._before_image
        if img:
            w_ratio = self.width() / img.width()
            h_ratio = self.height() / img.height()
            self._zoom = min(w_ratio, h_ratio)
            self._pan_offset = QPoint(0, 0)
            self.update()

    def toggle_before_after(self):
        """Swap split to show full before or full after."""
        self._split_pos = 0.0 if self._split_pos > 0.1 else 1.0
        self.update()

    # --- Ручные зоны лиц -----------------------------------------------------

    def set_zone_edit_mode(self, enabled: bool):
        self._zone_edit = bool(enabled)
        if not enabled:
            self._zone_drag = None
            self._sel_zone = None
        self.update()

    def set_zones(self, zones: list):
        self._zones = [list(zg.unpack_zone(z)) for z in (zones or [])]
        if self._sel_zone is not None and self._sel_zone >= len(self._zones):
            self._sel_zone = None
        self.update()

    def get_zones(self) -> list:
        return [list(z) for z in self._zones]

    def selected_zone(self):
        return self._sel_zone

    def delete_selected_zone(self):
        if self._sel_zone is not None and 0 <= self._sel_zone < len(self._zones):
            self._zones.pop(self._sel_zone)
            self._sel_zone = None
            self.zones_changed.emit(self.get_zones())
            self.update()

    def _zone_params(self):
        """Аргументы маппинга для zone_geometry (или None, если нет картинки)."""
        img = self._before_image or self._after_image
        if not img:
            return None
        return dict(widget_w=self.width(), widget_h=self.height(),
                    img_w=img.width(), img_h=img.height(),
                    zoom=self._zoom,
                    pan_x=self._pan_offset.x(), pan_y=self._pan_offset.y())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)

        img = self._after_image or self._before_image
        if not img:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Перетащите изображение или нажмите Загрузить")
            painter.end()
            return

        # Compute scaled image rect
        scaled_w = int(img.width() * self._zoom)
        scaled_h = int(img.height() * self._zoom)
        x = (self.width() - scaled_w) // 2 + self._pan_offset.x()
        y = (self.height() - scaled_h) // 2 + self._pan_offset.y()
        dest_rect = QRectF(x, y, scaled_w, scaled_h)

        # Draw before (left of splitter)
        split_x = int(self.width() * self._split_pos)

        if self._before_image:
            painter.setClipRect(0, 0, split_x, self.height())
            painter.drawImage(dest_rect, self._before_image)

        # Draw after (right of splitter)
        if self._after_image:
            painter.setClipRect(split_x, 0, self.width() - split_x, self.height())
            painter.drawImage(dest_rect, self._after_image)

        # Draw splitter line
        painter.setClipping(False)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawLine(split_x, 0, split_x, self.height())

        # Оверлей ручных зон лиц (режим разметки)
        params = self._zone_params()
        if self._zone_edit and params and self._zones:
            painter.setClipping(False)
            for i, zone in enumerate(self._zones):
                corners = [zg.norm_to_widget(cx, cy, **params)
                           for cx, cy in zg.zone_corners(zone)]
                selected = (i == self._sel_zone)
                color = QColor(80, 220, 100) if selected else QColor(240, 200, 60)
                pen = QPen(color)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(QColor(color.red(), color.green(),
                                        color.blue(), 40))
                painter.drawPolygon(QPolygonF([QPointF(px, py) for px, py in corners]))
                painter.setBrush(color)
                for cx, cy in corners:
                    painter.drawRect(int(cx) - 3, int(cy) - 3, 6, 6)
                x0, y0 = corners[0]
                painter.drawText(int(x0) + 4, int(y0) + 14, str(i + 1))
                if selected:
                    x, y, w, h, _angle = zg.unpack_zone(zone)
                    top_mid_nx, top_mid_ny = x + w / 2.0, y
                    tmx, tmy = zg.norm_to_widget(top_mid_nx, top_mid_ny, **params)
                    hnx, hny = zg.rotation_handle_norm(zone)
                    hx, hy = zg.norm_to_widget(hnx, hny, **params)
                    painter.drawLine(int(tmx), int(tmy), int(hx), int(hy))
                    painter.setBrush(QColor(80, 220, 100))
                    painter.drawEllipse(QPointF(hx, hy), 5, 5)

        painter.end()

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._zoom = max(0.1, min(self._zoom * factor, 20.0))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        params = self._zone_params()
        if self._zone_edit and params and \
                event.button() == Qt.MouseButton.LeftButton:
            px, py = event.position().x(), event.position().y()
            nx, ny = zg.widget_to_norm(px, py, **params)
            rot = zg.hit_test_rotation_handle(self._zones, px, py, **params)
            if rot is not None:
                self._sel_zone = rot
                self._zone_drag = "rotate"
                self._zone_orig = list(self._zones[rot])
                self._zone_anchor = (nx, ny)
                self.update()
                return
            handle = zg.hit_test_handle(self._zones, px, py, **params)
            if handle is not None:
                self._sel_zone, self._zone_drag_corner = handle
                self._zone_drag = "resize"
                self._zone_orig = list(self._zones[self._sel_zone])
            else:
                idx = zg.hit_test_zone(self._zones, nx, ny)
                if idx is not None:
                    self._sel_zone = idx
                    self._zone_drag = "move"
                    self._zone_orig = list(self._zones[idx])
                else:
                    self._zone_drag = "new"
                    self._sel_zone = None
            self._zone_anchor = (nx, ny)
            self.update()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            split_x = int(self.width() * self._split_pos)
            if abs(event.position().x() - split_x) < 10:
                self._dragging_splitter = True
            else:
                self._panning = True
                self._last_mouse_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        params = self._zone_params()
        if self._zone_edit and params and self._zone_drag:
            px, py = event.position().x(), event.position().y()
            nx, ny = zg.widget_to_norm(px, py, **params)
            ax, ay = self._zone_anchor
            if self._zone_drag == "move" and self._sel_zone is not None:
                self._zones[self._sel_zone] = zg.move_zone(
                    self._zone_orig, nx - ax, ny - ay)
            elif self._zone_drag == "resize" and self._sel_zone is not None:
                self._zones[self._sel_zone] = zg.resize_zone(
                    self._zone_orig, self._zone_drag_corner, nx, ny)
            elif self._zone_drag == "rotate" and self._sel_zone is not None:
                self._zones[self._sel_zone] = list(self._zone_orig[:4]) + [
                    zg.angle_from_handle_drag(self._zone_orig, nx, ny)]
            # "new": рамка рисуется по anchor..текущая в paintEvent через
            # временную зону — для простоты создаём/обновляем последнюю зону
            elif self._zone_drag == "new":
                rect = zg.rect_from_points(ax, ay, nx, ny) + [0.0]
                if self._zone_orig is None:
                    self._zones.append(rect)
                    self._zone_orig = "pending"
                else:
                    self._zones[-1] = rect
            self.update()
            return
        if self._dragging_splitter:
            self._split_pos = max(0.0, min(event.position().x() / self.width(), 1.0))
            self.update()
        elif self._panning:
            delta = event.position().toPoint() - self._last_mouse_pos
            self._pan_offset += delta
            self._last_mouse_pos = event.position().toPoint()
            self.update()
        else:
            split_x = int(self.width() * self._split_pos)
            if abs(event.position().x() - split_x) < 10:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._zone_edit and self._zone_drag:
            if self._zone_drag == "new":
                if self._zone_orig == "pending":
                    self._sel_zone = len(self._zones) - 1
                else:
                    # клик без движения: просто снять выделение
                    self._sel_zone = None
            self._zone_drag = None
            self._zone_orig = None
            self.zones_changed.emit(self.get_zones())
            self.update()
            return
        self._dragging_splitter = False
        self._panning = False
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if self._zone_edit and event.key() == Qt.Key.Key_Delete:
            self.delete_selected_zone()
            return
        super().keyPressEvent(event)

    def _ndarray_to_qimage(self, arr: np.ndarray) -> QImage:
        if arr.dtype != np.uint8:
            if arr.dtype in (np.float32, np.float64):
                arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            elif arr.dtype == np.uint16:
                arr = (arr / 257).astype(np.uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        h, w, c = arr.shape
        bytes_per_line = w * c
        if c == 3:
            return QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        elif c == 4:
            return QImage(arr.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        return QImage()
