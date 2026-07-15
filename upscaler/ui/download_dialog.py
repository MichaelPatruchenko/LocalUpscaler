"""Model download progress dialog."""

from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton
from PySide6.QtCore import Qt

from upscaler.ui.i18n import tr


class DownloadDialog(QDialog):
    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self.setWindowTitle(tr("download.title", model=model_name))
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        self.label = QLabel(tr("download.progress_label", model=model_name))
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.size_label = QLabel("")
        layout.addWidget(self.size_label)

        self.cancel_btn = QPushButton(tr("download.cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты диалога из текущего языка."""
        self.setWindowTitle(tr("download.title", model=self._model_name))
        self.label.setText(tr("download.progress_label", model=self._model_name))
        self.cancel_btn.setText(tr("download.cancel"))

    def update_progress(self, percent: int, downloaded: int, total: int):
        self.progress.setValue(percent)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        self.size_label.setText(f"{mb_done:.1f} MB / {mb_total:.1f} MB")
