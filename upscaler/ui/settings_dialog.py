"""Settings editor dialog with GPU detection and all configuration options."""

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QSpinBox,
    QLineEdit, QPushButton, QDialogButtonBox, QFileDialog,
    QGroupBox, QLabel, QHBoxLayout, QCheckBox,
)
from PySide6.QtCore import Qt

import torch

from upscaler.ui.i18n import tr


def _detect_gpu_options() -> list[tuple[str, object]]:
    """Return list of (value, kind) for GPU device options.

    kind is "cpu" for the CPU entry, or (index, name, mem_gb) for a CUDA
    device — the display label is built lazily via _gpu_option_text() so it
    can be recomputed on retranslate().
    """
    options = [("cpu", "cpu")]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            options.append((f"cuda:{i}", (i, name, mem)))
    return options


def _gpu_option_text(kind) -> str:
    if kind == "cpu":
        return "CPU"
    i, name, mem = kind
    return tr("settings.gpu_option", i=i, name=name, mem=f"{mem:.1f}")


class SettingsDialog(QDialog):
    def __init__(self, current_settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("settings.window_title"))
        self.setMinimumWidth(550)
        self._settings = dict(current_settings)
        self._gpu_kinds: dict[str, object] = {}  # combo value -> kind (for retranslate)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- GPU / Performance ---
        self.gpu_group = QGroupBox(tr("settings.gpu_group"))
        gpu_form = QFormLayout()

        self.cuda_info_label = QLabel(self._cuda_info_text())
        self.cuda_info_label.setStyleSheet("color: gray; font-size: 11px;")
        gpu_form.addRow(self.cuda_info_label)

        self.cuda_warn_label = None
        if not torch.cuda.is_available():
            self.cuda_warn_label = QLabel(tr("settings.cuda_warning"))
            self.cuda_warn_label.setStyleSheet("color: #cc6600; font-size: 11px;")
            self.cuda_warn_label.setWordWrap(True)
            gpu_form.addRow(self.cuda_warn_label)

        self.gpu_combo = QComboBox()
        gpu_options = _detect_gpu_options()
        current_device = self._settings.get("gpu_device", "auto")
        found_current = False
        for value, kind in gpu_options:
            self._gpu_kinds[value] = kind
            self.gpu_combo.addItem(_gpu_option_text(kind), value)
            if value == current_device:
                self.gpu_combo.setCurrentIndex(self.gpu_combo.count() - 1)
                found_current = True
        if not found_current:
            self.gpu_combo.insertItem(0, current_device, current_device)
            self.gpu_combo.setCurrentIndex(0)

        self.device_label = QLabel(tr("settings.device_label"))
        gpu_form.addRow(self.device_label, self.gpu_combo)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(128, 2048)
        self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(self._settings.get("tile_size", 512))
        self.tile_spin.setToolTip(tr("settings.tile_size_tooltip"))
        self.tile_size_label = QLabel(tr("settings.tile_size_label"))
        gpu_form.addRow(self.tile_size_label, self.tile_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(8, 128)
        self.overlap_spin.setSingleStep(8)
        self.overlap_spin.setValue(self._settings.get("tile_overlap", 32))
        self.overlap_spin.setToolTip(tr("settings.tile_overlap_tooltip"))
        self.tile_overlap_label = QLabel(tr("settings.tile_overlap_label"))
        gpu_form.addRow(self.tile_overlap_label, self.overlap_spin)

        self.gpu_denoise_cb = QCheckBox(tr("settings.gpu_denoise_checkbox"))
        self.gpu_denoise_cb.setChecked(
            self._settings.get("prefer_gpu_denoise", True))
        self.gpu_denoise_cb.setToolTip(tr("settings.gpu_denoise_tooltip"))
        gpu_form.addRow(self.gpu_denoise_cb)

        self.gpu_group.setLayout(gpu_form)
        layout.addWidget(self.gpu_group)

        # --- Output ---
        self.output_group = QGroupBox(tr("settings.output_group"))
        output_form = QFormLayout()

        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "jpg", "tiff", "exr"])
        self.format_combo.setCurrentText(self._settings.get("default_output_format", "png"))
        self.format_label = QLabel(tr("settings.format_label"))
        output_form.addRow(self.format_label, self.format_combo)

        dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self._settings.get("default_output_dir", ""))
        self.output_dir_edit.setPlaceholderText(tr("settings.output_dir_placeholder"))
        dir_row.addWidget(self.output_dir_edit)
        self.output_dir_browse_btn = QPushButton(tr("settings.browse"))
        self.output_dir_browse_btn.clicked.connect(self._browse_output_dir)
        dir_row.addWidget(self.output_dir_browse_btn)
        self.output_dir_label = QLabel(tr("settings.output_dir_label"))
        output_form.addRow(self.output_dir_label, dir_row)

        self.output_group.setLayout(output_form)
        layout.addWidget(self.output_group)

        # --- History ---
        self.history_group = QGroupBox(tr("settings.history_group"))
        history_form = QFormLayout()

        self.history_spin = QSpinBox()
        self.history_spin.setRange(5, 500)
        self.history_spin.setValue(self._settings.get("max_history_entries", 50))
        self.max_entries_label = QLabel(tr("settings.max_entries_label"))
        history_form.addRow(self.max_entries_label, self.history_spin)

        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(1, 90)
        self.retention_spin.setValue(self._settings.get("history_retention_days", 7))
        self.retention_spin.setSuffix(tr("settings.retention_suffix"))
        self.retention_label = QLabel(tr("settings.retention_label"))
        history_form.addRow(self.retention_label, self.retention_spin)

        self.history_group.setLayout(history_form)
        layout.addWidget(self.history_group)

        # --- Storage ---
        self.storage_group = QGroupBox(tr("settings.storage_group"))
        storage_form = QFormLayout()

        cache_row = QHBoxLayout()
        self.cache_dir_edit = QLineEdit(self._settings.get("model_cache_dir", ""))
        self.cache_dir_edit.setPlaceholderText(tr("settings.cache_dir_placeholder"))
        cache_row.addWidget(self.cache_dir_edit)
        self.cache_dir_browse_btn = QPushButton(tr("settings.browse"))
        self.cache_dir_browse_btn.clicked.connect(self._browse_cache_dir)
        cache_row.addWidget(self.cache_dir_browse_btn)
        self.cache_dir_label = QLabel(tr("settings.cache_dir_label"))
        storage_form.addRow(self.cache_dir_label, cache_row)

        self.storage_group.setLayout(storage_form)
        layout.addWidget(self.storage_group)

        # --- Appearance ---
        self.appearance_group = QGroupBox(tr("settings.appearance_group"))
        appearance_form = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["system", "light", "dark"])
        self.theme_combo.setCurrentText(self._settings.get("theme", "system"))
        self.theme_label = QLabel(tr("settings.theme_label"))
        appearance_form.addRow(self.theme_label, self.theme_combo)

        self.appearance_group.setLayout(appearance_form)
        layout.addWidget(self.appearance_group)

        # --- ICEdit ---
        self.icedit_group = QGroupBox(tr("settings.icedit_group"))
        icedit_form = QFormLayout()

        self.icedit_offload_combo = QComboBox()
        self.icedit_offload_combo.addItem(tr("settings.icedit_offload_model"), "model")
        self.icedit_offload_combo.addItem(tr("settings.icedit_offload_none"), "none")
        self.icedit_offload_combo.addItem(tr("settings.icedit_offload_sequential"), "sequential")
        idx = self.icedit_offload_combo.findData(self._settings.get("icedit_offload", "model"))
        self.icedit_offload_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.icedit_offload_label = QLabel(tr("settings.icedit_offload_label"))
        icedit_form.addRow(self.icedit_offload_label, self.icedit_offload_combo)

        self.icedit_quant_combo = QComboBox()
        self.icedit_quant_combo.addItem(tr("settings.icedit_quant_q4"), "q4")
        self.icedit_quant_combo.addItem(tr("settings.icedit_quant_q5"), "q5")
        idx = self.icedit_quant_combo.findData(self._settings.get("icedit_quant", "q4"))
        self.icedit_quant_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.icedit_quant_label = QLabel(tr("settings.icedit_quant_label"))
        icedit_form.addRow(self.icedit_quant_label, self.icedit_quant_combo)

        self.icedit_group.setLayout(icedit_form)
        layout.addWidget(self.icedit_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _cuda_info_text(self) -> str:
        status = tr("settings.cuda_available") if torch.cuda.is_available() \
            else tr("settings.cuda_unavailable")
        cuda_ver = torch.version.cuda or tr("settings.cuda_na")
        return tr("settings.info_text", ver=torch.__version__, status=status, cudaver=cuda_ver)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты диалога из текущего языка."""
        self.setWindowTitle(tr("settings.window_title"))
        self.gpu_group.setTitle(tr("settings.gpu_group"))
        self.cuda_info_label.setText(self._cuda_info_text())
        if self.cuda_warn_label is not None:
            self.cuda_warn_label.setText(tr("settings.cuda_warning"))
        for i in range(self.gpu_combo.count()):
            value = self.gpu_combo.itemData(i)
            if value in self._gpu_kinds:
                self.gpu_combo.setItemText(i, _gpu_option_text(self._gpu_kinds[value]))
        self.device_label.setText(tr("settings.device_label"))
        self.tile_size_label.setText(tr("settings.tile_size_label"))
        self.tile_spin.setToolTip(tr("settings.tile_size_tooltip"))
        self.tile_overlap_label.setText(tr("settings.tile_overlap_label"))
        self.overlap_spin.setToolTip(tr("settings.tile_overlap_tooltip"))
        self.gpu_denoise_cb.setText(tr("settings.gpu_denoise_checkbox"))
        self.gpu_denoise_cb.setToolTip(tr("settings.gpu_denoise_tooltip"))

        self.output_group.setTitle(tr("settings.output_group"))
        self.format_label.setText(tr("settings.format_label"))
        self.output_dir_edit.setPlaceholderText(tr("settings.output_dir_placeholder"))
        self.output_dir_browse_btn.setText(tr("settings.browse"))
        self.output_dir_label.setText(tr("settings.output_dir_label"))

        self.history_group.setTitle(tr("settings.history_group"))
        self.max_entries_label.setText(tr("settings.max_entries_label"))
        self.retention_spin.setSuffix(tr("settings.retention_suffix"))
        self.retention_label.setText(tr("settings.retention_label"))

        self.storage_group.setTitle(tr("settings.storage_group"))
        self.cache_dir_edit.setPlaceholderText(tr("settings.cache_dir_placeholder"))
        self.cache_dir_browse_btn.setText(tr("settings.browse"))
        self.cache_dir_label.setText(tr("settings.cache_dir_label"))

        self.appearance_group.setTitle(tr("settings.appearance_group"))
        self.theme_label.setText(tr("settings.theme_label"))

        self.icedit_group.setTitle(tr("settings.icedit_group"))
        self.icedit_offload_combo.setItemText(0, tr("settings.icedit_offload_model"))
        self.icedit_offload_combo.setItemText(1, tr("settings.icedit_offload_none"))
        self.icedit_offload_combo.setItemText(2, tr("settings.icedit_offload_sequential"))
        self.icedit_offload_label.setText(tr("settings.icedit_offload_label"))
        self.icedit_quant_combo.setItemText(0, tr("settings.icedit_quant_q4"))
        self.icedit_quant_combo.setItemText(1, tr("settings.icedit_quant_q5"))
        self.icedit_quant_label.setText(tr("settings.icedit_quant_label"))

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("settings.output_dir_dialog_title"))
        if d:
            self.output_dir_edit.setText(d)

    def _browse_cache_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("settings.cache_dir_dialog_title"))
        if d:
            self.cache_dir_edit.setText(d)

    def get_settings(self) -> dict:
        return {
            **self._settings,
            "gpu_device": self.gpu_combo.currentData(),
            "default_output_format": self.format_combo.currentText(),
            "default_output_dir": self.output_dir_edit.text(),
            "max_history_entries": self.history_spin.value(),
            "history_retention_days": self.retention_spin.value(),
            "tile_size": self.tile_spin.value(),
            "tile_overlap": self.overlap_spin.value(),
            "model_cache_dir": self.cache_dir_edit.text(),
            "theme": self.theme_combo.currentText(),
            "icedit_offload": self.icedit_offload_combo.currentData(),
            "icedit_quant": self.icedit_quant_combo.currentData(),
            "prefer_gpu_denoise": self.gpu_denoise_cb.isChecked(),
        }
