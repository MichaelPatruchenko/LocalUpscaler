"""Shared base: a thumbnail list with dual selection (primary/secondary).

Used by ``HistoryPanel`` (history.* i18n, revert/compare/delete buttons, info
group) and (upcoming) the variant list panel. This module owns the selection
state machine and thumbnail rendering only — it is i18n-neutral and has no
buttons of its own.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QColor, QBrush
import numpy as np

# Slot highlight colors.
_PRIMARY_BG = QColor(40, 90, 160)     # blue tint  → primary (left click)
_SECONDARY_BG = QColor(170, 95, 30)   # orange tint → secondary (right click)
_DEFAULT_BG = QColor(0, 0, 0, 0)      # transparent → unselected

# Custom item-data role used to remember the caller-supplied label (if any)
# so badges can be recomputed on every refresh without re-deriving it.
_LABEL_ROLE = Qt.ItemDataRole.UserRole + 1


class VersionListWidget(QListWidget):
    """QListWidget emitting separate signals for left/right clicks on items."""

    primary_clicked = Signal(int)
    secondary_clicked = Signal(int)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            version = int(item.data(Qt.ItemDataRole.UserRole))
            if event.button() == Qt.MouseButton.LeftButton:
                self.primary_clicked.emit(version)
                event.accept()
                return
            if event.button() == Qt.MouseButton.RightButton:
                self.secondary_clicked.emit(version)
                event.accept()
                return
        super().mousePressEvent(event)


class VersionListPanel(QWidget):
    selection_changed = Signal(int, int)  # primary, secondary (-1 = none)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._primary: int | None = None
        self._secondary: int | None = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.list_widget = self._build_list_widget()
        layout.addWidget(self.list_widget)

    def _build_list_widget(self) -> VersionListWidget:
        list_widget = VersionListWidget()
        list_widget.setIconSize(QSize(120, 90))
        list_widget.setSpacing(4)
        # We manage primary/secondary highlight ourselves; disable Qt's own
        # selection and context menu so they don't fight our model.
        list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        list_widget.primary_clicked.connect(self._on_primary_clicked)
        list_widget.secondary_clicked.connect(self._on_secondary_clicked)
        return list_widget

    # ─── Selection state machine (independent slots) ───
    #
    # The two slots are independent: clearing the primary does NOT pull the
    # secondary into the primary slot. The *effective* selection
    # (``_effective``) treats a lone secondary as the primary for display, the
    # working image, and the emitted signal. This preserves the required
    # behaviour: from a compare state (A=primary, B=secondary), left-clicking A
    # deselects it (B becomes the shown/working image), and a following
    # right-click on B clears it too → nothing selected.

    def _on_primary_clicked(self, version: int):
        if self._primary == version:
            self._primary = None  # deselect; leave the secondary slot untouched
        else:
            if self._secondary == version:
                self._secondary = None
            self._primary = version
        self._refresh()

    def _on_secondary_clicked(self, version: int):
        if self._secondary == version:
            self._secondary = None
        elif self._primary != version:
            self._secondary = version
        else:
            return  # right-click on the primary image: cannot occupy both slots
        self._refresh()

    def _effective(self) -> tuple[int | None, int | None]:
        """Normalize the two raw slots into (primary, secondary) for display.

        - both slots set → (primary, secondary)  [compare]
        - exactly one set → (that one, None)       [display; lone = primary]
        - neither set → (None, None)               [empty]
        """
        if self._primary is not None and self._secondary is not None:
            return self._primary, self._secondary
        lone = self._primary if self._primary is not None else self._secondary
        return lone, None

    def selected(self) -> tuple[int | None, int | None]:
        """Public view of the effective selection. The first element is the
        working image (used for Save and the next processing pass)."""
        return self._effective()

    def _refresh(self):
        eff_primary, eff_secondary = self._effective()
        self._update_badges(eff_primary, eff_secondary)
        self.selection_changed.emit(
            eff_primary if eff_primary is not None else -1,
            eff_secondary if eff_secondary is not None else -1,
        )

    def _badge_text(self, kind: str, id: int, base_label: str) -> str:
        """Overridable badge text. ``kind`` is "primary"/"secondary"/"none"."""
        if kind == "primary":
            return f"① v{id}"
        if kind == "secondary":
            return f"② v{id}"
        return base_label or f"v{id}"

    def _update_badges(self, eff_primary, eff_secondary):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            v = int(item.data(Qt.ItemDataRole.UserRole))
            base_label = item.data(_LABEL_ROLE) or ""
            if v == eff_primary:
                item.setText(self._badge_text("primary", v, base_label))
                item.setBackground(QBrush(_PRIMARY_BG))
            elif v == eff_secondary:
                item.setText(self._badge_text("secondary", v, base_label))
                item.setBackground(QBrush(_SECONDARY_BG))
            else:
                item.setText(self._badge_text("none", v, base_label))
                item.setBackground(QBrush(_DEFAULT_BG))

    # ─── Public API ───

    def add_item(self, id: int, thumbnail: np.ndarray | None, label: str = ""):
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, int(id))
        item.setData(_LABEL_ROLE, label)
        if thumbnail is not None:
            qimg = self._ndarray_to_qimage(thumbnail)
            item.setIcon(QIcon(QPixmap.fromImage(qimg)))
        self.list_widget.insertItem(0, item)
        # A freshly added item becomes the active (primary) selection.
        self._primary = int(id)
        self._secondary = None
        self._refresh()

    def clear(self):
        self.list_widget.clear()
        self._primary = None
        self._secondary = None
        # Do NOT emit: callers (e.g. image load) set the canvas themselves.

    def remove_item(self, id: int):
        for i in range(self.list_widget.count()):
            if int(self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)) == id:
                self.list_widget.takeItem(i)
                break
        # Clear whichever slot held the removed id.
        if self._primary == id:
            self._primary = None
        if self._secondary == id:
            self._secondary = None
        self._refresh()

    def _ndarray_to_qimage(self, arr: np.ndarray) -> QImage:
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
        h, w, c = arr.shape
        return QImage(arr.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
