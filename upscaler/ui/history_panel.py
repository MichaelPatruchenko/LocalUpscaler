"""Right panel: history thumbnails with dual selection (primary/secondary)."""

from PySide6.QtWidgets import (
    QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QGroupBox,
)
from PySide6.QtCore import Signal
import numpy as np

from upscaler.ui.i18n import tr
from upscaler.ui.version_list_panel import VersionListPanel


class HistoryPanel(VersionListPanel):
    revert_requested = Signal(int)
    compare_requested = Signal(int)
    delete_requested = Signal(int)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.title_label = QLabel(tr("history.title"))
        layout.addWidget(self.title_label)

        self.list_widget = self._build_list_widget()
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.revert_btn = QPushButton(tr("history.revert"))
        self.revert_btn.clicked.connect(self._on_revert)
        btn_layout.addWidget(self.revert_btn)
        self.compare_btn = QPushButton(tr("history.compare"))
        self.compare_btn.clicked.connect(self._on_compare)
        btn_layout.addWidget(self.compare_btn)
        self.delete_btn = QPushButton(tr("history.delete"))
        self.delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.delete_btn)
        layout.addLayout(btn_layout)

        self.info_group = QGroupBox(tr("history.info_group"))
        info_layout = QVBoxLayout()
        self.info_label = QLabel(tr("history.info_none"))
        info_layout.addWidget(self.info_label)
        self.info_group.setLayout(info_layout)
        layout.addWidget(self.info_group)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.title_label.setText(tr("history.title"))
        self.revert_btn.setText(tr("history.revert"))
        self.compare_btn.setText(tr("history.compare"))
        self.delete_btn.setText(tr("history.delete"))
        self.info_group.setTitle(tr("history.info_group"))
        self.info_label.setText(tr("history.info_none"))

    # ─── Public API ───

    def add_version(self, version: int, thumbnail: np.ndarray | None, metrics: dict):
        self.add_item(version, thumbnail)

    def clear(self):
        super().clear()
        self.info_label.setText(tr("history.info_none"))

    # ─── Buttons (operate on the primary slot) ───

    def _on_revert(self):
        eff_primary, _ = self._effective()
        if eff_primary is not None:
            self.revert_requested.emit(eff_primary)

    def _on_compare(self):
        eff_primary, _ = self._effective()
        if eff_primary is not None:
            self.compare_requested.emit(eff_primary)

    def _on_delete(self):
        eff_primary, _ = self._effective()
        if eff_primary is None:
            return
        version = eff_primary
        self.delete_requested.emit(version)
        self.remove_item(version)
