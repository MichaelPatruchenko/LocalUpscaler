"""Batch processing panel with image thumbnails and dual processing modes."""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QLabel,
    QProgressBar, QFileDialog, QHBoxLayout,
)
from PySide6.QtCore import Signal, QSize, Qt
from PySide6.QtGui import QImage, QPixmap, QIcon
import numpy as np
import cv2

from upscaler.ui.i18n import tr


class BatchPanel(QWidget):
    process_batch_requested = Signal()
    beautiful_batch_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._files: list[Path] = []
        self._output_dir = ""
        self._last_progress = None  # (current, total, filename) or None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        btn_layout = QHBoxLayout()
        self.add_files_btn = QPushButton(tr("batch.add_files"))
        self.add_files_btn.clicked.connect(self._on_add_files)
        btn_layout.addWidget(self.add_files_btn)

        self.add_dir_btn = QPushButton(tr("batch.add_dir"))
        self.add_dir_btn.clicked.connect(self._on_add_dir)
        btn_layout.addWidget(self.add_dir_btn)

        self.clear_btn = QPushButton(tr("batch.clear"))
        self.clear_btn.clicked.connect(self._on_clear)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        # Thumbnail list
        self.file_list = QListWidget()
        self.file_list.setIconSize(QSize(150, 150))
        self.file_list.setSpacing(4)
        self.file_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.file_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.file_list.setWrapping(True)
        layout.addWidget(self.file_list)

        self.file_count_label = QLabel(tr("batch.file_count", n=0))
        layout.addWidget(self.file_count_label)

        # Output dir
        out_layout = QHBoxLayout()
        self.output_caption_label = QLabel(tr("batch.output_caption"))
        out_layout.addWidget(self.output_caption_label)
        self.output_label = QLabel(tr("batch.output_default"))
        out_layout.addWidget(self.output_label, stretch=1)
        self.output_btn = QPushButton(tr("batch.browse"))
        self.output_btn.clicked.connect(self._on_browse_output)
        out_layout.addWidget(self.output_btn)
        layout.addLayout(out_layout)

        # Two processing buttons
        proc_layout = QHBoxLayout()

        self.process_btn = QPushButton(tr("batch.process_btn"))
        self.process_btn.setToolTip(tr("batch.process_tooltip"))
        self.process_btn.setStyleSheet("QPushButton { padding: 8px; font-weight: bold; }")
        self.process_btn.clicked.connect(self.process_batch_requested)
        proc_layout.addWidget(self.process_btn)

        self.beautiful_btn = QPushButton(tr("batch.beautiful_btn"))
        self.beautiful_btn.setToolTip(tr("batch.beautiful_tooltip"))
        self.beautiful_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #6b21a8; color: white; }"
            "QPushButton:hover { background-color: #7c3aed; }"
        )
        self.beautiful_btn.clicked.connect(self.beautiful_batch_requested)
        proc_layout.addWidget(self.beautiful_btn)

        layout.addLayout(proc_layout)

        # Progress
        self.batch_progress = QLabel("")
        layout.addWidget(self.batch_progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.add_files_btn.setText(tr("batch.add_files"))
        self.add_dir_btn.setText(tr("batch.add_dir"))
        self.clear_btn.setText(tr("batch.clear"))
        self.file_count_label.setText(tr("batch.file_count", n=len(self._files)))
        self.output_caption_label.setText(tr("batch.output_caption"))
        if not self._output_dir:
            self.output_label.setText(tr("batch.output_default"))
        self.output_btn.setText(tr("batch.browse"))
        self.process_btn.setText(tr("batch.process_btn"))
        self.process_btn.setToolTip(tr("batch.process_tooltip"))
        self.beautiful_btn.setText(tr("batch.beautiful_btn"))
        self.beautiful_btn.setToolTip(tr("batch.beautiful_tooltip"))
        if self._last_progress is not None:
            current, total, filename = self._last_progress
            self.batch_progress.setText(
                tr("batch.progress", current=current, total=total, filename=filename))

    def _generate_thumbnail(self, path: Path) -> QIcon | None:
        try:
            data = path.read_bytes()
            arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                return None
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            h, w = arr.shape[:2]
            scale = 150 / max(h, w)
            thumb = cv2.resize(arr, (int(w * scale), int(h * scale)))
            qimg = QImage(thumb.tobytes(), thumb.shape[1], thumb.shape[0],
                          thumb.shape[1] * 3, QImage.Format.Format_RGB888)
            return QIcon(QPixmap.fromImage(qimg))
        except Exception:
            return None

    def _add_file(self, path: Path):
        self._files.append(path)
        item = QListWidgetItem(path.name)
        icon = self._generate_thumbnail(path)
        if icon:
            item.setIcon(icon)
        self.file_list.addItem(item)

    def _on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("batch.select_images_title"), "",
            tr("batch.images_filter")
        )
        for f in files:
            self._add_file(Path(f))
        self.file_count_label.setText(tr("batch.file_count", n=len(self._files)))

    def _on_add_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("batch.select_dir_title"))
        if d:
            self.add_directory(d)

    def add_directory(self, path: str):
        from upscaler.config import SUPPORTED_INPUT_FORMATS
        for f in sorted(Path(path).iterdir()):
            if f.suffix.lower() in SUPPORTED_INPUT_FORMATS:
                self._add_file(f)
        self.file_count_label.setText(tr("batch.file_count", n=len(self._files)))

    def _on_browse_output(self):
        d = QFileDialog.getExistingDirectory(self, tr("batch.output_dir_title"))
        if d:
            self._output_dir = d
            self.output_label.setText(d)

    def _on_clear(self):
        self._files.clear()
        self.file_list.clear()
        self.file_count_label.setText(tr("batch.file_count", n=0))

    def get_files(self) -> list[Path]:
        return list(self._files)

    def get_output_dir(self) -> str:
        return self._output_dir

    def set_progress(self, current: int, total: int, filename: str):
        self._last_progress = (current, total, filename)
        self.batch_progress.setText(
            tr("batch.progress", current=current, total=total, filename=filename))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(current / total * 100))
