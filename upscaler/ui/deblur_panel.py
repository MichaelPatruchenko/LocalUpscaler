"""SmartDeblur fine-tuning panel: auto/manual deconvolution controls."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
    QGroupBox, QSlider, QRadioButton, QButtonGroup, QCheckBox,
)
from PySide6.QtCore import Qt, Signal

from upscaler.ui.i18n import tr


class DeblurPanel(QWidget):
    deblur_params_changed = Signal(dict)
    preview_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slider_labels: list[tuple[QLabel, str]] = []  # (caption widget, i18n key)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title_label = QLabel(tr("deblur.title"))
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Mode
        self.mode_group_box = QGroupBox(tr("deblur.mode_group"))
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.auto_rb = QRadioButton(tr("deblur.mode_auto"))
        self.manual_rb = QRadioButton(tr("deblur.mode_manual"))
        self.auto_rb.setChecked(True)
        self.mode_group.addButton(self.auto_rb)
        self.mode_group.addButton(self.manual_rb)
        self.auto_rb.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self.auto_rb)
        mode_layout.addWidget(self.manual_rb)
        self.mode_group_box.setLayout(mode_layout)
        layout.addWidget(self.mode_group_box)

        # Manual controls
        self.manual_group = QGroupBox(tr("deblur.params_group"))
        man_layout = QVBoxLayout()

        self.blur_type_label = QLabel(tr("deblur.blur_type_label"))
        man_layout.addWidget(self.blur_type_label)
        self.blur_type_combo = QComboBox()
        self.blur_type_combo.addItem(tr("deblur.blur_focus"), "focus")
        self.blur_type_combo.addItem(tr("deblur.blur_motion"), "motion")
        self.blur_type_combo.addItem(tr("deblur.blur_gaussian"), "gaussian")
        self.blur_type_combo.currentIndexChanged.connect(self._on_blur_type_changed)
        man_layout.addWidget(self.blur_type_combo)

        self.radius_slider, self.radius_row = self._slider_row(
            man_layout, "deblur.radius", 1, 500, 30, 10.0)
        self.angle_slider, self.angle_row = self._slider_row(
            man_layout, "deblur.angle", 0, 180, 0, 1.0)
        self.smooth_slider, self.smooth_row = self._slider_row(
            man_layout, "deblur.smooth", 1, 100, 30, 1.0)
        self.feather_slider, self.feather_row = self._slider_row(
            man_layout, "deblur.feather", 0, 100, 10, 1.0)
        self.correction_slider, self.correction_row = self._slider_row(
            man_layout, "deblur.correction", 0, 100, 0, 1.0)

        self.method_label = QLabel(tr("deblur.method_label"))
        man_layout.addWidget(self.method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItem(tr("deblur.method_wiener"), "wiener")
        self.method_combo.addItem(tr("deblur.method_tikhonov"), "tikhonov")
        self.method_combo.addItem(tr("deblur.method_tv"), "tv")
        self.method_combo.addItem(tr("deblur.method_rl"), "rl")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        man_layout.addWidget(self.method_combo)

        self.tv_slider, self.tv_row = self._slider_row(
            man_layout, "deblur.iterations", 10, 1000, 300, 1.0)

        self.edge_taper_cb = QCheckBox(tr("deblur.edge_taper"))
        self.edge_taper_cb.setChecked(True)
        self.edge_taper_cb.toggled.connect(self._emit_changed)
        man_layout.addWidget(self.edge_taper_cb)

        self.manual_group.setLayout(man_layout)
        layout.addWidget(self.manual_group)

        self.preview_btn = QPushButton(tr("deblur.preview_btn"))
        self.preview_btn.clicked.connect(
            lambda: self.preview_requested.emit(self.get_deblur_config()))
        layout.addWidget(self.preview_btn)

        self._on_mode_changed()
        self._on_blur_type_changed()
        self._on_method_changed()

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.title_label.setText(tr("deblur.title"))
        self.mode_group_box.setTitle(tr("deblur.mode_group"))
        self.auto_rb.setText(tr("deblur.mode_auto"))
        self.manual_rb.setText(tr("deblur.mode_manual"))
        self.manual_group.setTitle(tr("deblur.params_group"))
        self.blur_type_label.setText(tr("deblur.blur_type_label"))
        self.blur_type_combo.setItemText(0, tr("deblur.blur_focus"))
        self.blur_type_combo.setItemText(1, tr("deblur.blur_motion"))
        self.blur_type_combo.setItemText(2, tr("deblur.blur_gaussian"))
        for caption, key in self._slider_labels:
            caption.setText(f"{tr(key)}:")
        self.method_label.setText(tr("deblur.method_label"))
        self.method_combo.setItemText(0, tr("deblur.method_wiener"))
        self.method_combo.setItemText(1, tr("deblur.method_tikhonov"))
        self.method_combo.setItemText(2, tr("deblur.method_tv"))
        self.method_combo.setItemText(3, tr("deblur.method_rl"))
        self.edge_taper_cb.setText(tr("deblur.edge_taper"))
        self.preview_btn.setText(tr("deblur.preview_btn"))

    def _slider_row(self, layout, key, mn, mx, default, scale):
        row = QHBoxLayout()
        caption = QLabel(f"{tr(key)}:")
        self._slider_labels.append((caption, key))
        row.addWidget(caption)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(default)
        slider.setProperty("scale", scale)
        value_label = QLabel(f"{default / scale:.1f}")
        slider.valueChanged.connect(
            lambda v, lbl=value_label, s=scale: lbl.setText(f"{v / s:.1f}"))
        slider.valueChanged.connect(self._emit_changed)
        row.addWidget(slider)
        row.addWidget(value_label)
        container = QWidget()
        container.setLayout(row)
        layout.addWidget(container)
        return slider, container

    def _on_mode_changed(self, *_):
        self.manual_group.setEnabled(self.manual_rb.isChecked())
        self._emit_changed()

    def _on_blur_type_changed(self, *_):
        bt = self.blur_type_combo.currentData()
        self.angle_row.setVisible(bt == "motion")
        self.feather_row.setVisible(bt == "focus")
        self.correction_row.setVisible(bt == "focus")
        self._emit_changed()

    def _on_method_changed(self, *_):
        # iterations apply to the iterative methods (TV and Richardson-Lucy)
        self.tv_row.setVisible(self.method_combo.currentData() in ("tv", "rl"))
        self._emit_changed()

    def set_auto(self, is_auto: bool):
        self.auto_rb.setChecked(is_auto)
        self.manual_rb.setChecked(not is_auto)

    def set_blur_type(self, blur_type: str):
        idx = self.blur_type_combo.findData(blur_type)
        if idx >= 0:
            self.blur_type_combo.setCurrentIndex(idx)

    def _slider_value(self, slider):
        return slider.value() / float(slider.property("scale"))

    def get_deblur_config(self) -> dict:
        if self.auto_rb.isChecked():
            return {"auto": True, "method": self.method_combo.currentData(),
                    "edge_taper": self.edge_taper_cb.isChecked()}
        return {
            "auto": False,
            "blur_type": self.blur_type_combo.currentData(),
            "radius": self._slider_value(self.radius_slider),
            "angle": self._slider_value(self.angle_slider),
            "smooth": self._slider_value(self.smooth_slider),
            "edge_feather": self._slider_value(self.feather_slider),
            "correction_strength": self._slider_value(self.correction_slider),
            "method": self.method_combo.currentData(),
            "tv_iterations": int(self._slider_value(self.tv_slider)),
            "edge_taper": self.edge_taper_cb.isChecked(),
        }

    def apply_config(self, cfg: dict):
        """Reflect an externally-chosen deblur config (auto/LLM) on the controls.

        Signals are blocked so this does not re-emit and overwrite the config
        that was just stored on the control panel.
        """
        if not cfg:
            return
        self.blockSignals(True)
        try:
            self.set_auto(bool(cfg.get("auto", True)))
            if cfg.get("blur_type"):
                self.set_blur_type(cfg["blur_type"])

            def _set(slider, key):
                if key in cfg:
                    scale = float(slider.property("scale"))
                    slider.setValue(int(round(float(cfg[key]) * scale)))

            _set(self.radius_slider, "radius")
            _set(self.angle_slider, "angle")
            _set(self.smooth_slider, "smooth")
            _set(self.feather_slider, "edge_feather")
            _set(self.correction_slider, "correction_strength")
            _set(self.tv_slider, "tv_iterations")

            if cfg.get("method"):
                idx = self.method_combo.findData(cfg["method"])
                if idx >= 0:
                    self.method_combo.setCurrentIndex(idx)
            if "edge_taper" in cfg:
                self.edge_taper_cb.setChecked(bool(cfg["edge_taper"]))

            self._on_blur_type_changed()
            self._on_method_changed()
        finally:
            self.blockSignals(False)

    def _emit_changed(self, *_):
        self.deblur_params_changed.emit(self.get_deblur_config())
