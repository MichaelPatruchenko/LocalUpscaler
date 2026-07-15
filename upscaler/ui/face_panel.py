# upscaler/ui/face_panel.py
"""CodeFormer face-restoration fine-tuning panel."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox,
    QSlider, QCheckBox, QListWidget,
)
from PySide6.QtCore import Qt, Signal

from upscaler.ui.i18n import tr


class FacePanel(QWidget):
    face_params_changed = Signal(dict)
    preview_requested = Signal(dict)
    zone_edit_toggled = Signal(bool)
    zones_edited = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zones: list = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title_label = QLabel(tr("face.title"))
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.params_group = QGroupBox(tr("face.params_group"))
        g = QVBoxLayout()

        # Fidelity slider (0..100 -> 0.0..1.0)
        row = QHBoxLayout()
        self.fidelity_caption_label = QLabel(tr("face.fidelity_label"))
        row.addWidget(self.fidelity_caption_label)
        self.fidelity_slider = QSlider(Qt.Orientation.Horizontal)
        self.fidelity_slider.setRange(0, 100)
        self.fidelity_slider.setValue(70)
        self.fid_label = QLabel("0.70")
        self.fidelity_slider.valueChanged.connect(
            lambda v: self.fid_label.setText(f"{v / 100:.2f}"))
        self.fidelity_slider.valueChanged.connect(self._emit_changed)
        row.addWidget(self.fidelity_slider)
        row.addWidget(self.fid_label)
        g.addLayout(row)

        self.fidelity_hint_label = QLabel(tr("face.fidelity_hint"))
        self.fidelity_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.fidelity_hint_label.setWordWrap(True)
        g.addWidget(self.fidelity_hint_label)

        # min_face_px slider
        row2 = QHBoxLayout()
        self.minpx_caption_label = QLabel(tr("face.minpx_label"))
        row2.addWidget(self.minpx_caption_label)
        self.minpx_slider = QSlider(Qt.Orientation.Horizontal)
        self.minpx_slider.setRange(16, 256)
        self.minpx_slider.setValue(32)
        self.minpx_label = QLabel("32")
        self.minpx_slider.valueChanged.connect(
            lambda v: self.minpx_label.setText(str(v)))
        self.minpx_slider.valueChanged.connect(self._emit_changed)
        row2.addWidget(self.minpx_slider)
        row2.addWidget(self.minpx_label)
        g.addLayout(row2)

        self.bg_cb = QCheckBox(tr("face.bg_checkbox"))
        self.bg_cb.setChecked(False)
        self.bg_cb.toggled.connect(self._emit_changed)
        g.addWidget(self.bg_cb)

        self.params_group.setLayout(g)
        layout.addWidget(self.params_group)

        self.zones_group = QGroupBox(tr("face.zones_group"))
        zg_layout = QVBoxLayout()
        self.zone_edit_btn = QPushButton(tr("face.zone_edit_btn"))
        self.zone_edit_btn.setCheckable(True)
        self.zone_edit_btn.toggled.connect(self.zone_edit_toggled.emit)
        zg_layout.addWidget(self.zone_edit_btn)
        self.zones_list = QListWidget()
        self.zones_list.setFixedHeight(110)
        zg_layout.addWidget(self.zones_list)
        btn_row = QHBoxLayout()
        self.delete_zone_btn = QPushButton(tr("face.delete_zone"))
        self.clear_zones_btn = QPushButton(tr("face.clear_zones"))
        self.delete_zone_btn.clicked.connect(self._delete_selected_zone)
        self.clear_zones_btn.clicked.connect(self._clear_zones)
        btn_row.addWidget(self.delete_zone_btn)
        btn_row.addWidget(self.clear_zones_btn)
        zg_layout.addLayout(btn_row)
        self.zones_hint_label = QLabel(tr("face.zones_hint"))
        self.zones_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.zones_hint_label.setWordWrap(True)
        zg_layout.addWidget(self.zones_hint_label)
        self.zones_group.setLayout(zg_layout)
        layout.addWidget(self.zones_group)

        self.preview_btn = QPushButton(tr("face.preview_btn"))
        self.preview_btn.clicked.connect(
            lambda: self.preview_requested.emit(self.get_face_config()))
        layout.addWidget(self.preview_btn)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.title_label.setText(tr("face.title"))
        self.params_group.setTitle(tr("face.params_group"))
        self.fidelity_caption_label.setText(tr("face.fidelity_label"))
        self.fidelity_hint_label.setText(tr("face.fidelity_hint"))
        self.minpx_caption_label.setText(tr("face.minpx_label"))
        self.bg_cb.setText(tr("face.bg_checkbox"))
        self.zones_group.setTitle(tr("face.zones_group"))
        self.zone_edit_btn.setText(tr("face.zone_edit_btn"))
        self.delete_zone_btn.setText(tr("face.delete_zone"))
        self.clear_zones_btn.setText(tr("face.clear_zones"))
        self.zones_hint_label.setText(tr("face.zones_hint"))
        self.preview_btn.setText(tr("face.preview_btn"))
        self._refresh_zone_list()

    def get_face_config(self) -> dict:
        return {
            "enabled": True,
            "fidelity": self.fidelity_slider.value() / 100.0,
            "min_face_px": self.minpx_slider.value(),
            "upscale_background": self.bg_cb.isChecked(),
            "regions": self.get_zones(),
        }

    def apply_config(self, cfg: dict):
        if not cfg:
            return
        self.blockSignals(True)
        try:
            if "fidelity" in cfg:
                self.fidelity_slider.setValue(int(round(float(cfg["fidelity"]) * 100)))
            if "min_face_px" in cfg:
                self.minpx_slider.setValue(int(cfg["min_face_px"]))
            if "upscale_background" in cfg:
                self.bg_cb.setChecked(bool(cfg["upscale_background"]))
            if "regions" in cfg:
                self._zones = [list(z) for z in (cfg["regions"] or [])]
                self._refresh_zone_list()
        finally:
            self.blockSignals(False)

    def _emit_changed(self, *_):
        self.face_params_changed.emit(self.get_face_config())

    # --- Зоны лиц ------------------------------------------------------------

    def set_zones(self, zones: list):
        """Обновление зон от холста (или программно)."""
        self._zones = [list(z) for z in (zones or [])]
        self._refresh_zone_list()
        self._emit_changed()

    def get_zones(self) -> list:
        return [list(z) for z in self._zones]

    def _refresh_zone_list(self):
        self.zones_list.clear()
        for i, zone in enumerate(self._zones):
            x, y, w, h = zone[0], zone[1], zone[2], zone[3]
            angle = float(zone[4]) if len(zone) > 4 else 0.0
            caption = tr("face.zone_item", i=i + 1, w=f"{w * 100:.0f}", h=f"{h * 100:.0f}",
                   x=f"{x * 100:.0f}", y=f"{y * 100:.0f}")
            if angle != 0.0:
                caption += f", {angle:.0f}°"
            self.zones_list.addItem(caption)

    def _delete_selected_zone(self):
        row = self.zones_list.currentRow()
        if 0 <= row < len(self._zones):
            self._zones.pop(row)
            self._refresh_zone_list()
            self.zones_edited.emit(self.get_zones())
            self._emit_changed()

    def _clear_zones(self):
        if self._zones:
            self._zones = []
            self._refresh_zone_list()
            self.zones_edited.emit([])
            self._emit_changed()

    def clear_zones(self):
        """Публичная очистка зон (используется тулбаром главного экрана)."""
        self._clear_zones()
