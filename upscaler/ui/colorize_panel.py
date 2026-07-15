"""Colorization panel: photo/video colorization with DDColor, DeOldify, ColorMNet."""

from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
    QGroupBox, QSlider, QProgressBar, QFileDialog,
)
from PySide6.QtCore import Qt, Signal

from upscaler.ui.i18n import tr


class ColorizePanel(QWidget):
    colorize_photo_requested = Signal(str, dict)  # model_name, params
    colorize_video_requested = Signal(str, dict, str, str)  # model, params, input_path, output_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: str = ""
        self._last_video_progress = None  # (current, total) or None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Title
        self.title_label = QLabel(tr("colorize.title"))
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Model selection
        self.model_group = QGroupBox(tr("colorize.model_group"))
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItem(tr("colorize.model_ddcolor"), "DDColor")
        self.model_combo.addItem(tr("colorize.model_deoldify"), "DeOldify")
        self.model_combo.addItem(tr("colorize.model_colormnet"), "ColorMNet")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)

        # DDColor variant selector
        self.ddcolor_variant_row = QWidget()
        ddcolor_row_layout = QHBoxLayout(self.ddcolor_variant_row)
        ddcolor_row_layout.setContentsMargins(0, 0, 0, 0)
        self.ddcolor_variant_label = QLabel(tr("colorize.variant_label"))
        ddcolor_row_layout.addWidget(self.ddcolor_variant_label)
        self.ddcolor_variant_combo = QComboBox()
        self.ddcolor_variant_combo.addItem(tr("colorize.ddcolor_artistic"), "artistic")
        self.ddcolor_variant_combo.addItem(tr("colorize.ddcolor_modelscope"), "modelscope")
        ddcolor_row_layout.addWidget(self.ddcolor_variant_combo)
        model_layout.addWidget(self.ddcolor_variant_row)

        # DeOldify variant selector
        self.deoldify_variant_row = QWidget()
        deoldify_row_layout = QHBoxLayout(self.deoldify_variant_row)
        deoldify_row_layout.setContentsMargins(0, 0, 0, 0)
        self.deoldify_variant_label = QLabel(tr("colorize.variant_label"))
        deoldify_row_layout.addWidget(self.deoldify_variant_label)
        self.deoldify_variant_combo = QComboBox()
        self.deoldify_variant_combo.addItem(tr("colorize.deoldify_stable"), "stable")
        self.deoldify_variant_combo.addItem(tr("colorize.deoldify_artistic"), "artistic")
        self.deoldify_variant_combo.addItem(tr("colorize.deoldify_video"), "video")
        deoldify_row_layout.addWidget(self.deoldify_variant_combo)
        model_layout.addWidget(self.deoldify_variant_row)
        self.deoldify_variant_row.setVisible(False)

        # Model info label
        self.model_info = QLabel("")
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("color: gray; font-size: 11px;")
        model_layout.addWidget(self.model_info)
        self._update_model_info()

        self.model_group.setLayout(model_layout)
        layout.addWidget(self.model_group)

        # Parameters
        self.params_group = QGroupBox(tr("colorize.params_group"))
        params_layout = QVBoxLayout()

        # Strength slider
        strength_row = QHBoxLayout()
        self.strength_caption_label = QLabel(tr("colorize.strength_label"))
        strength_row.addWidget(self.strength_caption_label)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_label = QLabel("1.00")
        self.strength_slider.valueChanged.connect(
            lambda v: self.strength_label.setText(f"{v / 100:.2f}")
        )
        strength_row.addWidget(self.strength_slider)
        strength_row.addWidget(self.strength_label)
        params_layout.addLayout(strength_row)

        self.params_group.setLayout(params_layout)
        layout.addWidget(self.params_group)

        # Photo section
        self.photo_group = QGroupBox(tr("colorize.photo_group"))
        photo_layout = QVBoxLayout()

        self.colorize_photo_btn = QPushButton(tr("colorize.photo_btn"))
        self.colorize_photo_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #b45309; color: white; }"
            "QPushButton:hover { background-color: #d97706; }"
        )
        self.colorize_photo_btn.clicked.connect(self._on_colorize_photo)
        photo_layout.addWidget(self.colorize_photo_btn)

        self.photo_group.setLayout(photo_layout)
        layout.addWidget(self.photo_group)

        # Video section
        self.video_group = QGroupBox(tr("colorize.video_group"))
        video_layout = QVBoxLayout()

        self.video_path_label = QLabel(tr("colorize.no_video"))
        video_layout.addWidget(self.video_path_label)

        vid_btn_row = QHBoxLayout()
        self.select_video_btn = QPushButton(tr("colorize.select_video"))
        self.select_video_btn.clicked.connect(self._on_select_video)
        vid_btn_row.addWidget(self.select_video_btn)

        self.colorize_video_btn = QPushButton(tr("colorize.video_btn"))
        self.colorize_video_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #b45309; color: white; }"
            "QPushButton:hover { background-color: #d97706; }"
        )
        self.colorize_video_btn.clicked.connect(self._on_colorize_video)
        self.colorize_video_btn.setEnabled(False)
        vid_btn_row.addWidget(self.colorize_video_btn)
        video_layout.addLayout(vid_btn_row)

        # Video progress
        self.video_progress = QLabel("")
        video_layout.addWidget(self.video_progress)
        self.video_progress_bar = QProgressBar()
        self.video_progress_bar.setVisible(False)
        video_layout.addWidget(self.video_progress_bar)

        self.video_group.setLayout(video_layout)
        layout.addWidget(self.video_group)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка."""
        self.title_label.setText(tr("colorize.title"))
        self.model_group.setTitle(tr("colorize.model_group"))
        self.model_combo.setItemText(0, tr("colorize.model_ddcolor"))
        self.model_combo.setItemText(1, tr("colorize.model_deoldify"))
        self.model_combo.setItemText(2, tr("colorize.model_colormnet"))
        self.ddcolor_variant_label.setText(tr("colorize.variant_label"))
        self.ddcolor_variant_combo.setItemText(0, tr("colorize.ddcolor_artistic"))
        self.ddcolor_variant_combo.setItemText(1, tr("colorize.ddcolor_modelscope"))
        self.deoldify_variant_label.setText(tr("colorize.variant_label"))
        self.deoldify_variant_combo.setItemText(0, tr("colorize.deoldify_stable"))
        self.deoldify_variant_combo.setItemText(1, tr("colorize.deoldify_artistic"))
        self.deoldify_variant_combo.setItemText(2, tr("colorize.deoldify_video"))
        self._update_model_info()
        self.params_group.setTitle(tr("colorize.params_group"))
        self.strength_caption_label.setText(tr("colorize.strength_label"))
        self.photo_group.setTitle(tr("colorize.photo_group"))
        self.colorize_photo_btn.setText(tr("colorize.photo_btn"))
        self.video_group.setTitle(tr("colorize.video_group"))
        if not self._video_path:
            self.video_path_label.setText(tr("colorize.no_video"))
        self.select_video_btn.setText(tr("colorize.select_video"))
        self.colorize_video_btn.setText(tr("colorize.video_btn"))
        if self._last_video_progress is not None:
            current, total = self._last_video_progress
            self.video_progress.setText(
                tr("colorize.frame_progress", current=current, total=total))

    def _on_model_changed(self, index):
        self._update_model_info()
        model_name = self.model_combo.currentData()
        # Show/hide variant selectors
        self.ddcolor_variant_row.setVisible(model_name == "DDColor")
        self.deoldify_variant_row.setVisible(model_name == "DeOldify")
        # DDColor: photo only — disable video
        supports_video = model_name in ("DeOldify", "ColorMNet")
        self.video_group.setEnabled(supports_video)

    def _update_model_info(self):
        model = self.model_combo.currentData()
        info = {
            "DDColor": tr("colorize.info_ddcolor"),
            "DeOldify": tr("colorize.info_deoldify"),
            "ColorMNet": tr("colorize.info_colormnet"),
        }
        self.model_info.setText(info.get(model, ""))

    def get_colorize_params(self) -> dict:
        model = self.model_combo.currentData()
        params = {
            "strength": self.strength_slider.value() / 100.0,
        }
        if model == "DDColor":
            params["variant"] = self.ddcolor_variant_combo.currentData()
        elif model == "DeOldify":
            params["variant"] = self.deoldify_variant_combo.currentData()
        return params

    def get_selected_model(self) -> str:
        return self.model_combo.currentData()

    def _on_colorize_photo(self):
        model = self.get_selected_model()
        params = self.get_colorize_params()
        self.colorize_photo_requested.emit(model, params)

    def _on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("colorize.select_video"), "",
            tr("colorize.video_filter")
        )
        if path:
            self._video_path = path
            self.video_path_label.setText(Path(path).name)
            self.colorize_video_btn.setEnabled(True)

    def _on_colorize_video(self):
        if not self._video_path:
            return
        model = self.get_selected_model()
        if model == "DDColor":
            return  # DDColor doesn't support video

        params = self.get_colorize_params()
        output_path, _ = QFileDialog.getSaveFileName(
            self, tr("colorize.save_video_title"), "",
            tr("colorize.video_save_filter")
        )
        if output_path:
            self.colorize_video_requested.emit(model, params, self._video_path, output_path)

    def set_video_progress(self, current: int, total: int):
        self._last_video_progress = (current, total)
        self.video_progress.setText(tr("colorize.frame_progress", current=current, total=total))
        self.video_progress_bar.setVisible(True)
        self.video_progress_bar.setValue(int(current / total * 100))
