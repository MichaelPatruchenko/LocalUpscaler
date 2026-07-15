# upscaler/ui/blend_panel.py
"""Вкладка «Смешивание»: композиция вариантов конвейера (режимы Photoshop).

Чисто-UI виджет: показывает список вариантов (снимки шагов текущего
прогона + результаты смешивания), даёт выбрать пару первичный/вторичный
(как в HistoryPanel), режим наложения и прозрачность, эмитит сигналы —
резолв изображений и композиция остаются на стороне MainWindow.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSlider,
)

from upscaler.engine.blend import BLEND_MODES, blend_mode_label
from upscaler.ui.i18n import tr
from upscaler.ui.version_list_panel import VersionListPanel


class VariantPanel(VersionListPanel):
    """VersionListPanel без кнопок истории — просто подписанный список.

    Наполняется извне (``add_item``/``clear``, унаследованные), используется
    внутри ``BlendPanel`` как ``variant_list``.
    """

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = QLabel(tr("blend.variants"))
        layout.addWidget(self.title_label)
        self.list_widget = self._build_list_widget()
        layout.addWidget(self.list_widget)

    def retranslate(self) -> None:
        self.title_label.setText(tr("blend.variants"))


class BlendPanel(QWidget):
    # {"primary": id, "secondary": id, "mode": str, "opacity": float}
    blend_selected_requested = Signal(dict)
    preview_requested = Signal(dict)
    apply_requested = Signal(dict)
    auto_requested = Signal()
    selection_changed = Signal(int, int)  # primary, secondary (-1 = none)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title_label = QLabel(tr("blend.panel_title"))
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.variant_list = VariantPanel()
        self.variant_list.setFixedHeight(220)
        self.variant_list.selection_changed.connect(self.selection_changed.emit)
        layout.addWidget(self.variant_list)

        row_mode = QHBoxLayout()
        self.mode_caption_label = QLabel(tr("blend.mode_caption"))
        row_mode.addWidget(self.mode_caption_label)
        self.mode_combo = QComboBox()
        for mode in BLEND_MODES:
            self.mode_combo.addItem(blend_mode_label(mode), mode)
        row_mode.addWidget(self.mode_combo)
        layout.addLayout(row_mode)

        row_opacity = QHBoxLayout()
        self.opacity_caption_label = QLabel(tr("blend.opacity_caption"))
        row_opacity.addWidget(self.opacity_caption_label)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(50)
        self.opacity_label = QLabel("50%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%"))
        row_opacity.addWidget(self.opacity_slider)
        row_opacity.addWidget(self.opacity_label)
        layout.addLayout(row_opacity)

        act = QHBoxLayout()
        self.blend_selected_btn = QPushButton(tr("blend.blend_selected"))
        self.blend_selected_btn.clicked.connect(self._emit_blend_selected)
        self.auto_btn = QPushButton(tr("blend.auto_btn"))
        self.auto_btn.clicked.connect(self.auto_requested.emit)
        self.preview_btn = QPushButton(tr("blend.preview_btn"))
        self.preview_btn.clicked.connect(self._emit_preview)
        self.apply_btn = QPushButton(tr("blend.apply_btn"))
        self.apply_btn.clicked.connect(self._emit_apply)
        act.addWidget(self.blend_selected_btn)
        act.addWidget(self.auto_btn)
        act.addWidget(self.preview_btn)
        act.addWidget(self.apply_btn)
        layout.addLayout(act)

        self.hint_label = QLabel(tr("blend.hint"))
        self.hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.title_label.setText(tr("blend.panel_title"))
        if hasattr(self.variant_list, "retranslate"):
            self.variant_list.retranslate()
        self.mode_caption_label.setText(tr("blend.mode_caption"))
        for i in range(self.mode_combo.count()):
            mode = self.mode_combo.itemData(i)
            self.mode_combo.setItemText(i, blend_mode_label(mode))
        self.opacity_caption_label.setText(tr("blend.opacity_caption"))
        self.blend_selected_btn.setText(tr("blend.blend_selected"))
        self.auto_btn.setText(tr("blend.auto_btn"))
        self.preview_btn.setText(tr("blend.preview_btn"))
        self.apply_btn.setText(tr("blend.apply_btn"))
        self.hint_label.setText(tr("blend.hint"))

    # --- Варианты ---------------------------------------------------------

    def set_variants(self, variants) -> None:
        """Заменить содержимое списка. *variants* — итерируемое из
        ``(id, thumbnail, label)`` (или dict с теми же ключами)."""
        self.variant_list.clear()
        for item in variants:
            if isinstance(item, dict):
                self.add_variant(item["id"], item.get("thumbnail"),
                                 item.get("label", ""))
            else:
                variant_id, thumbnail, label = item
                self.add_variant(variant_id, thumbnail, label)

    def add_variant(self, id, thumbnail, label: str = "") -> None:
        self.variant_list.add_item(id, thumbnail, label)

    # --- Рецепт (пара выбранных вариантов + режим/прозрачность) -----------

    def get_recipe(self) -> dict | None:
        """Текущий рецепт по выбору в списке, либо ``None`` — если выбрана
        не пара (нужны ровно первичный и вторичный варианты)."""
        primary, secondary = self.variant_list.selected()
        if primary is None or secondary is None:
            return None
        return {
            "primary": primary,
            "secondary": secondary,
            "mode": self.mode_combo.currentData(),
            "opacity": self.opacity_slider.value() / 100.0,
        }

    def _emit_blend_selected(self):
        recipe = self.get_recipe()
        if recipe is not None:
            self.blend_selected_requested.emit(recipe)

    def _emit_preview(self):
        recipe = self.get_recipe()
        if recipe is not None:
            self.preview_requested.emit(recipe)

    def _emit_apply(self):
        recipe = self.get_recipe()
        if recipe is not None:
            self.apply_requested.emit(recipe)
