"""ICEdit panel: type a natural-language instruction and preview the edit."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
    QGroupBox, QSlider, QPlainTextEdit, QSpinBox,
)
from PySide6.QtCore import Qt, Signal

from upscaler.ui.i18n import tr


class ICEditPanel(QWidget):
    icedit_params_changed = Signal(dict)
    preview_requested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._slider_labels: list[tuple[QLabel, str]] = []  # (caption widget, i18n key)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.title_label = QLabel(tr("icedit.title"))
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        self.instruction_caption_label = QLabel(tr("icedit.instruction_label"))
        layout.addWidget(self.instruction_caption_label)
        self.instruction_edit = QPlainTextEdit()
        self.instruction_edit.setPlaceholderText(tr("icedit.instruction_placeholder"))
        self.instruction_edit.setFixedHeight(70)
        self.instruction_edit.textChanged.connect(self._emit_changed)
        layout.addWidget(self.instruction_edit)

        self.params_group = QGroupBox(tr("icedit.params_group"))
        man_layout = QVBoxLayout()

        self.lora_label = QLabel(tr("icedit.lora_label"))
        man_layout.addWidget(self.lora_label)
        self.variant_combo = QComboBox()
        self.variant_combo.addItem(tr("icedit.lora_normal"), "normal")
        self.variant_combo.addItem(tr("icedit.lora_moe"), "moe")
        self.variant_combo.currentIndexChanged.connect(self._emit_changed)
        man_layout.addWidget(self.variant_combo)

        self.steps_slider, _ = self._slider_row(man_layout, "icedit.steps_label", 8, 50, 28)
        self.guidance_slider, _ = self._slider_row(
            man_layout, "icedit.guidance_label", 1, 60, 50)

        seed_row = QHBoxLayout()
        self.seed_label = QLabel(tr("icedit.seed_label"))
        seed_row.addWidget(self.seed_label)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(-1)
        self.seed_spin.valueChanged.connect(self._emit_changed)
        seed_row.addWidget(self.seed_spin)
        man_layout.addLayout(seed_row)

        self.quant_label = QLabel(tr("icedit.quant_label"))
        man_layout.addWidget(self.quant_label)
        self.quant_combo = QComboBox()
        self.quant_combo.addItem(tr("icedit.quant_q4"), "q4")
        self.quant_combo.addItem(tr("icedit.quant_q5"), "q5")
        self.quant_combo.currentIndexChanged.connect(self._emit_changed)
        man_layout.addWidget(self.quant_combo)

        self.vram_label = QLabel(tr("icedit.vram_label"))
        man_layout.addWidget(self.vram_label)
        self.offload_combo = QComboBox()
        self.offload_combo.addItem(tr("icedit.offload_model"), "model")
        self.offload_combo.addItem(tr("icedit.offload_none"), "none")
        self.offload_combo.addItem(tr("icedit.offload_sequential"), "sequential")
        self.offload_combo.currentIndexChanged.connect(self._emit_changed)
        man_layout.addWidget(self.offload_combo)

        self.params_group.setLayout(man_layout)
        layout.addWidget(self.params_group)

        self.preview_btn = QPushButton(tr("icedit.preview_btn"))
        self.preview_btn.clicked.connect(
            lambda: self.preview_requested.emit(self.get_icedit_config()))
        layout.addWidget(self.preview_btn)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.title_label.setText(tr("icedit.title"))
        self.instruction_caption_label.setText(tr("icedit.instruction_label"))
        self.instruction_edit.setPlaceholderText(tr("icedit.instruction_placeholder"))
        self.params_group.setTitle(tr("icedit.params_group"))
        self.lora_label.setText(tr("icedit.lora_label"))
        self.variant_combo.setItemText(0, tr("icedit.lora_normal"))
        self.variant_combo.setItemText(1, tr("icedit.lora_moe"))
        for caption, key in self._slider_labels:
            caption.setText(f"{tr(key)}:")
        self.seed_label.setText(tr("icedit.seed_label"))
        self.quant_label.setText(tr("icedit.quant_label"))
        self.quant_combo.setItemText(0, tr("icedit.quant_q4"))
        self.quant_combo.setItemText(1, tr("icedit.quant_q5"))
        self.vram_label.setText(tr("icedit.vram_label"))
        self.offload_combo.setItemText(0, tr("icedit.offload_model"))
        self.offload_combo.setItemText(1, tr("icedit.offload_none"))
        self.offload_combo.setItemText(2, tr("icedit.offload_sequential"))
        self.preview_btn.setText(tr("icedit.preview_btn"))

    def _slider_row(self, layout, key, mn, mx, default):
        row = QHBoxLayout()
        caption = QLabel(f"{tr(key)}:")
        self._slider_labels.append((caption, key))
        row.addWidget(caption)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(mn, mx)
        slider.setValue(default)
        value_label = QLabel(str(default))
        slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
        slider.valueChanged.connect(self._emit_changed)
        row.addWidget(slider)
        row.addWidget(value_label)
        container = QWidget()
        container.setLayout(row)
        layout.addWidget(container)
        return slider, container

    def set_instruction(self, text: str):
        self.instruction_edit.setPlainText(text)

    def get_icedit_config(self) -> dict:
        instruction = self.instruction_edit.toPlainText().strip()
        return {
            "enabled": bool(instruction),
            "instruction": instruction,
            "variant": self.variant_combo.currentData(),
            "steps": self.steps_slider.value(),
            "guidance": float(self.guidance_slider.value()),
            "seed": self.seed_spin.value(),
            "quant": self.quant_combo.currentData(),
            "offload": self.offload_combo.currentData(),
        }

    def apply_config(self, cfg: dict):
        """Reflect an externally-chosen ICEdit config (auto/LLM) on the controls."""
        if not cfg:
            return
        self.blockSignals(True)
        try:
            if "instruction" in cfg:
                self.instruction_edit.setPlainText(cfg["instruction"] or "")
            if cfg.get("variant"):
                idx = self.variant_combo.findData(cfg["variant"])
                if idx >= 0:
                    self.variant_combo.setCurrentIndex(idx)
            if "steps" in cfg:
                self.steps_slider.setValue(int(cfg["steps"]))
            if "guidance" in cfg:
                self.guidance_slider.setValue(int(cfg["guidance"]))
            if "seed" in cfg:
                self.seed_spin.setValue(int(cfg["seed"]))
            if cfg.get("quant"):
                idx = self.quant_combo.findData(cfg["quant"])
                if idx >= 0:
                    self.quant_combo.setCurrentIndex(idx)
            if cfg.get("offload"):
                idx = self.offload_combo.findData(cfg["offload"])
                if idx >= 0:
                    self.offload_combo.setCurrentIndex(idx)
        finally:
            self.blockSignals(False)

    def _emit_changed(self, *_):
        self.icedit_params_changed.emit(self.get_icedit_config())
