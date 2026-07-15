"""Main application window: 3-panel layout, menu bar, status bar, action toolbar."""

import logging
import tempfile
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QMenuBar, QStatusBar, QFileDialog, QMessageBox, QLabel, QProgressBar,
    QStackedWidget, QRadioButton, QPushButton, QToolButton, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence

from upscaler.ui.canvas_widget import BeforeAfterCanvas, compute_canvas_images
from upscaler.ui.control_panel import ControlPanel
from upscaler.ui.history_panel import HistoryPanel
from upscaler.ui.batch_panel import BatchPanel
from upscaler.ui.colorize_panel import ColorizePanel
from upscaler.ui.deblur_panel import DeblurPanel
from upscaler.ui.icedit_panel import ICEditPanel
from upscaler.ui.face_panel import FacePanel
from upscaler.ui.blend_panel import BlendPanel
from upscaler.engine.controller import ProcessingController
from upscaler.engine.blend import blend
from upscaler.history.manager import HistoryManager
from upscaler.history.variant_store import VariantStore
from upscaler.utils.image_io import read_image, write_image
from upscaler.config import SUPPORTED_INPUT_FORMATS, load_settings, save_settings
from upscaler.ui.i18n import tr, translator, current_language, set_language

import cv2
import numpy as np

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("win.title"))
        self.setMinimumSize(1200, 800)

        self.settings = load_settings()
        self.controller = ProcessingController(self)
        self.history_manager = HistoryManager(
            max_entries=self.settings.get("max_history_entries", 50)
        )
        # Blend tab (Block G): variant snapshots live in their own session,
        # independent from the persistent history.
        self.variant_store = VariantStore()
        self.variant_store.create_session()
        self._current_image: np.ndarray | None = None
        self._original_image: np.ndarray | None = None
        self._current_meta: dict = {}
        self._current_path: Path | None = None
        self._llm_advisor = None  # lazily created LLMAdvisor
        self._refine_thread = None  # active LLMRefineThread, if any
        self._eval_thread = None   # active LLMEvaluateThread, if any
        self._iter_state = None     # active refinement loop state, if any
        self._last_blend_result = None  # last manual blend preview/apply result

        # Batch state
        self._batch_queue: list[Path] = []
        self._batch_mode: str = ""
        self._batch_output_dir: str = ""
        self._batch_index: int = 0
        self._batch_current_source: Path | None = None

        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        self._connect_signals()
        self._setup_status_bar()

        # Check for existing sessions and offer resume
        sessions = self.history_manager.list_sessions()
        resumed = False
        if sessions:
            reply = QMessageBox.question(
                self, tr("msg.resume_session_title"),
                tr("msg.resume_session_body", n=len(sessions)),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                resumed = self.history_manager.resume_session(sessions[0]["id"])
                if resumed:
                    count = self.history_manager.get_version_count()
                    if count > 0:
                        self._current_image = self.history_manager.get_version_image(count)
                        for v in range(1, count + 1):
                            thumb = self.history_manager.get_thumbnail(v)
                            meta = self.history_manager.get_metadata(v) or {}
                            self.history_panel.add_version(v, thumb, meta)
            else:
                # User declined — delete all previous sessions
                for s in sessions:
                    self.history_manager.delete_session(s["id"])

        if not resumed:
            self.history_manager.create_session()

        self.history_manager.cleanup_old_sessions(self.settings.get("history_retention_days", 7))
        self.controller.start_engine()

        geo = self.settings.get("window_geometry")
        restored = False
        if geo:
            try:
                from PySide6.QtCore import QByteArray
                restored = self.restoreGeometry(
                    QByteArray.fromBase64(geo.encode("ascii")))
            except Exception:
                restored = False
        if not restored:
            self.setWindowState(Qt.WindowState.WindowMaximized)

        # i18n: реагировать на переключение языка и применить persisted язык
        # (settings["language"]) к уже построенному UI один раз при старте.
        translator().language_changed.connect(lambda _l: self._retranslate_all())
        self._retranslate_all()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Top area: 3-panel splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: mode toggle + stacked panels
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        mode_layout = QHBoxLayout()
        self.single_mode_rb = QRadioButton(tr("mode.single"))
        self.single_mode_rb.setChecked(True)
        self.batch_mode_rb = QRadioButton(tr("mode.batch"))
        self.colorize_mode_rb = QRadioButton(tr("mode.colorize"))
        self.deblur_mode_rb = QRadioButton(tr("mode.deblur"))
        self.icedit_mode_rb = QRadioButton(tr("mode.icedit"))
        self.face_mode_rb = QRadioButton(tr("mode.face"))
        self.blend_mode_rb = QRadioButton(tr("mode.blend"))
        mode_layout.addWidget(self.single_mode_rb)
        mode_layout.addWidget(self.batch_mode_rb)
        mode_layout.addWidget(self.colorize_mode_rb)
        mode_layout.addWidget(self.deblur_mode_rb)
        mode_layout.addWidget(self.icedit_mode_rb)
        mode_layout.addWidget(self.face_mode_rb)
        mode_layout.addWidget(self.blend_mode_rb)
        left_layout.addLayout(mode_layout)

        self.left_stack = QStackedWidget()
        self.control_panel = ControlPanel()
        self.control_panel.set_max_refine_iterations(
            self.settings.get("max_refine_iterations", 3))
        self.control_panel.max_iter_changed.connect(self._on_max_iter_changed)
        self.control_panel.set_predownscale_enabled(
            bool(self.settings.get("auto_predownscale", True)))
        self.control_panel.predownscale_toggled.connect(
            self._on_predownscale_toggled)
        self.control_panel.set_section_states(
            self.settings.get("panel_sections", {}))
        self.control_panel.section_toggled.connect(self._on_section_toggled)
        self.batch_panel = BatchPanel()
        self.colorize_panel = ColorizePanel()
        self.deblur_panel = DeblurPanel()
        self.icedit_panel = ICEditPanel()
        self.face_panel = FacePanel()
        self.blend_panel = BlendPanel()
        self.left_stack.addWidget(self.control_panel)
        self.left_stack.addWidget(self.batch_panel)
        self.left_stack.addWidget(self.colorize_panel)
        self.left_stack.addWidget(self.deblur_panel)
        self.left_stack.addWidget(self.icedit_panel)
        self.left_stack.addWidget(self.face_panel)
        self.left_stack.addWidget(self.blend_panel)
        left_layout.addWidget(self.left_stack)

        self.single_mode_rb.toggled.connect(lambda checked: self._switch_mode(0) if checked else None)
        self.batch_mode_rb.toggled.connect(lambda checked: self._switch_mode(1) if checked else None)
        self.colorize_mode_rb.toggled.connect(lambda checked: self._switch_mode(2) if checked else None)
        self.deblur_mode_rb.toggled.connect(lambda checked: self._switch_mode(3) if checked else None)
        self.icedit_mode_rb.toggled.connect(lambda checked: self._switch_mode(4) if checked else None)
        self.face_mode_rb.toggled.connect(lambda checked: self._switch_mode(5) if checked else None)
        self.blend_mode_rb.toggled.connect(lambda checked: self._switch_mode(6) if checked else None)

        left_container.setFixedWidth(360)
        splitter.addWidget(left_container)

        # Center: тулбар зон лиц + холст
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(2)
        self._zones_bar = QWidget()  # видима только в одиночном режиме
        zones_bar = QHBoxLayout(self._zones_bar)
        zones_bar.setContentsMargins(0, 0, 0, 0)
        self.zones_toolbar_btn = QToolButton()
        self.zones_toolbar_btn.setText(tr("zones.toolbar_label"))
        self.zones_toolbar_btn.setCheckable(True)
        self.zones_toolbar_btn.setToolTip(tr("zones.toolbar_tooltip"))
        self.zones_clear_btn = QPushButton(tr("zones.clear_btn"))
        self._zone_count = 0
        self.zones_count_label = QLabel(tr("status.zone_count", n=0))
        zones_bar.addWidget(self.zones_toolbar_btn)
        zones_bar.addWidget(self.zones_clear_btn)
        zones_bar.addWidget(self.zones_count_label)
        # Полоса зон занимает по ширине только свои элементы (не весь холст)
        # и минимальную высоту, не отодвигая изображение.
        self._zones_bar.setSizePolicy(QSizePolicy.Policy.Maximum,
                                      QSizePolicy.Policy.Fixed)
        zones_row = QHBoxLayout()
        zones_row.setContentsMargins(0, 0, 0, 0)
        zones_row.addWidget(self._zones_bar)
        zones_row.addStretch(1)
        center_layout.addLayout(zones_row)
        self.canvas = BeforeAfterCanvas()
        center_layout.addWidget(self.canvas)
        splitter.addWidget(center_container)

        # Right: history
        self.history_panel = HistoryPanel()
        self.history_panel.setFixedWidth(280)
        splitter.addWidget(self.history_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        main_layout.addWidget(splitter, stretch=1)

        # Bottom: action buttons toolbar (always visible under canvas)
        btn_toolbar = QWidget()
        btn_layout = QHBoxLayout(btn_toolbar)
        btn_layout.setContentsMargins(0, 4, 0, 0)

        self.upload_btn = QPushButton(tr("btn.upload_image"))
        self.upload_btn.setMinimumHeight(36)
        btn_layout.addWidget(self.upload_btn)

        self.upload_dir_btn = QPushButton(tr("btn.upload_dir"))
        self.upload_dir_btn.setMinimumHeight(36)
        btn_layout.addWidget(self.upload_dir_btn)

        self.auto_btn = QPushButton(tr("btn.auto"))
        self.auto_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #2d7d46; color: white; }"
            "QPushButton:hover { background-color: #35a254; }"
        )
        self.auto_btn.setMinimumHeight(36)
        self.auto_btn.setToolTip(tr("tooltip.auto_btn"))
        btn_layout.addWidget(self.auto_btn)

        self.make_beautiful_btn = QPushButton(tr("btn.make_beautiful"))
        self.make_beautiful_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; background-color: #6b21a8; color: white; }"
            "QPushButton:hover { background-color: #7c3aed; }"
        )
        self.make_beautiful_btn.setMinimumHeight(36)
        self.make_beautiful_btn.setToolTip(tr("tooltip.make_beautiful_btn"))
        btn_layout.addWidget(self.make_beautiful_btn)

        self.process_btn = QPushButton(tr("btn.process"))
        self.process_btn.setStyleSheet(
            "QPushButton { padding: 8px; font-weight: bold; font-size: 13px; }"
        )
        self.process_btn.setMinimumHeight(36)
        btn_layout.addWidget(self.process_btn)

        self.lang_btn = QToolButton()
        self.lang_btn.setText(current_language().upper())
        self.lang_btn.setToolTip(tr("lang.toggle_tooltip"))
        self.lang_btn.setMinimumHeight(36)
        self.lang_btn.clicked.connect(self._on_toggle_language)
        btn_layout.addWidget(self.lang_btn)

        main_layout.addWidget(btn_toolbar)

    def _switch_mode(self, index: int):
        self.left_stack.setCurrentIndex(index)
        # Тулбар зон лиц относится к одиночному режиму (индекс 0).
        self._zones_bar.setVisible(index == 0)

    @Slot()
    def _on_toggle_language(self):
        """Переключить язык интерфейса на противоположный (RU <-> EN)."""
        new_lang = "en" if current_language() == "ru" else "ru"
        set_language(new_lang)

    def retranslate(self) -> None:
        """Переустановить собственные видимые тексты окна из текущего языка."""
        self.setWindowTitle(tr("win.title"))
        self.upload_btn.setText(tr("btn.upload_image"))
        self.upload_dir_btn.setText(tr("btn.upload_dir"))
        self.auto_btn.setText(tr("btn.auto"))
        self.auto_btn.setToolTip(tr("tooltip.auto_btn"))
        self.make_beautiful_btn.setText(tr("btn.make_beautiful"))
        self.make_beautiful_btn.setToolTip(tr("tooltip.make_beautiful_btn"))
        self.process_btn.setText(tr("btn.process"))
        self.single_mode_rb.setText(tr("mode.single"))
        self.batch_mode_rb.setText(tr("mode.batch"))
        self.colorize_mode_rb.setText(tr("mode.colorize"))
        self.deblur_mode_rb.setText(tr("mode.deblur"))
        self.icedit_mode_rb.setText(tr("mode.icedit"))
        self.face_mode_rb.setText(tr("mode.face"))
        self.blend_mode_rb.setText(tr("mode.blend"))
        self.lang_btn.setText(current_language().upper())
        self.lang_btn.setToolTip(tr("lang.toggle_tooltip"))

        # Тулбар зон лиц
        self.zones_toolbar_btn.setText(tr("zones.toolbar_label"))
        self.zones_toolbar_btn.setToolTip(tr("zones.toolbar_tooltip"))
        self.zones_clear_btn.setText(tr("zones.clear_btn"))
        self.zones_count_label.setText(tr("status.zone_count", n=self._zone_count))

        # Меню
        self.file_menu.setTitle(tr("menu.file"))
        self.open_action.setText(tr("menu.open"))
        self.save_action.setText(tr("menu.save"))
        self.exit_action.setText(tr("menu.exit"))
        self.edit_menu.setTitle(tr("menu.edit"))
        self.settings_action.setText(tr("menu.settings"))
        self.clear_history_action.setText(tr("menu.clear_history"))
        self.view_menu.setTitle(tr("menu.view"))
        self.fit_action.setText(tr("menu.fit_view"))
        self.proc_menu.setTitle(tr("menu.processing"))
        self.process_action.setText(tr("menu.process_action"))
        self.cancel_action.setText(tr("menu.cancel_action"))
        self.help_menu.setTitle(tr("menu.help"))
        self.about_action.setText(tr("menu.about"))

        # Статус-бар: переустановить последний показанный статус в новом языке.
        key, kwargs = getattr(self, "_last_status", ("status.ready", {}))
        self.status_label.setText(tr(key, **kwargs))
        self._update_gpu_label()
        self._update_disk_usage()

    def _set_zone_count(self, n: int) -> None:
        """Обновить счётчик зон лиц в тулбаре (запоминает значение для retranslate())."""
        self._zone_count = n
        self.zones_count_label.setText(tr("status.zone_count", n=n))

    def _set_status(self, key: str, **kwargs) -> None:
        """Установить текст статус-бара по ключу i18n, запомнив его для
        последующей ретрансляции при переключении языка."""
        self._last_status = (key, kwargs)
        self.status_label.setText(tr(key, **kwargs))

    def _retranslate_all(self) -> None:
        """Ретранслировать окно и все дочерние панели/диалоги, у которых есть
        метод retranslate() (не у всех он ещё реализован — hasattr-guard)."""
        self.retranslate()
        panels = (
            self.control_panel, self.batch_panel, self.colorize_panel,
            self.deblur_panel, self.icedit_panel, self.face_panel,
            self.blend_panel, self.history_panel,
        )
        for panel in panels:
            if hasattr(panel, "retranslate"):
                panel.retranslate()

    def _setup_menu(self):
        menubar = self.menuBar()

        # File
        self.file_menu = menubar.addMenu(tr("menu.file"))
        self.open_action = QAction(tr("menu.open"), self)
        self.open_action.triggered.connect(self._on_open_image)
        self.file_menu.addAction(self.open_action)

        self.save_action = QAction(tr("menu.save"), self)
        self.save_action.triggered.connect(self._on_save_result)
        self.file_menu.addAction(self.save_action)

        self.file_menu.addSeparator()
        self.exit_action = QAction(tr("menu.exit"), self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        # Edit
        self.edit_menu = menubar.addMenu(tr("menu.edit"))
        self.settings_action = QAction(tr("menu.settings"), self)
        self.settings_action.triggered.connect(self._on_settings)
        self.edit_menu.addAction(self.settings_action)

        self.clear_history_action = QAction(tr("menu.clear_history"), self)
        self.clear_history_action.triggered.connect(self._on_clear_history)
        self.edit_menu.addAction(self.clear_history_action)

        # View
        self.view_menu = menubar.addMenu(tr("menu.view"))
        self.fit_action = QAction(tr("menu.fit_view"), self)
        self.fit_action.triggered.connect(self.canvas.fit_to_view)
        self.view_menu.addAction(self.fit_action)

        # Processing
        self.proc_menu = menubar.addMenu(tr("menu.processing"))
        self.process_action = QAction(tr("menu.process_action"), self)
        self.process_action.triggered.connect(self._on_process)
        self.proc_menu.addAction(self.process_action)

        self.cancel_action = QAction(tr("menu.cancel_action"), self)
        self.cancel_action.triggered.connect(self._on_cancel)
        self.proc_menu.addAction(self.cancel_action)

        # Help
        self.help_menu = menubar.addMenu(tr("menu.help"))
        self.about_action = QAction(tr("menu.about"), self)
        self.about_action.triggered.connect(lambda: QMessageBox.about(
            self, tr("msg.about_title"), tr("msg.about_body")
        ))
        self.help_menu.addAction(self.about_action)

    def _setup_shortcuts(self):
        from PySide6.QtGui import QShortcut
        QShortcut(QKeySequence("Ctrl+O"), self, self._on_open_image)
        QShortcut(QKeySequence("Ctrl+P"), self, self._on_process)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._on_revert_to_original)
        QShortcut(QKeySequence("Space"), self, self.canvas.toggle_before_after)
        QShortcut(QKeySequence("Ctrl+0"), self, self.canvas.fit_to_view)
        QShortcut(QKeySequence("+"), self, lambda: self.canvas.set_zoom(self.canvas._zoom * 1.2))
        QShortcut(QKeySequence("-"), self, lambda: self.canvas.set_zoom(self.canvas._zoom / 1.2))

    def _connect_signals(self):
        # Toolbar buttons
        self.process_btn.clicked.connect(self._on_process)
        self.upload_btn.clicked.connect(self._on_open_image)
        self.upload_dir_btn.clicked.connect(self._on_open_dir)
        self.auto_btn.clicked.connect(self._on_auto_config)
        self.make_beautiful_btn.clicked.connect(self._on_make_beautiful)

        # History panel
        self.history_panel.revert_requested.connect(self._on_revert_to_original)
        self.history_panel.compare_requested.connect(self._on_compare)
        self.history_panel.selection_changed.connect(self._on_history_selection)
        self.history_panel.delete_requested.connect(self._on_history_delete)

        # Batch panel
        self.batch_panel.process_batch_requested.connect(self._on_batch_process)
        self.batch_panel.beautiful_batch_requested.connect(self._on_batch_beautiful)

        # Colorize panel
        self.colorize_panel.colorize_photo_requested.connect(self._on_colorize_photo)
        self.colorize_panel.colorize_video_requested.connect(self._on_colorize_video)

        # Deblur (SmartDeblur) panel
        self.deblur_panel.deblur_params_changed.connect(self.control_panel.set_deblur_config)
        self.deblur_panel.preview_requested.connect(self._on_deblur_preview)

        # ICEdit panel
        self.icedit_panel.icedit_params_changed.connect(self.control_panel.set_icedit_config)
        self.icedit_panel.preview_requested.connect(self._on_icedit_preview)

        # Face (CodeFormer) panel
        self.face_panel.face_params_changed.connect(self.control_panel.set_face_config)
        self.face_panel.preview_requested.connect(self._on_face_preview)
        self.face_panel.zones_edited.connect(self.canvas.set_zones)
        self.canvas.zones_changed.connect(self.face_panel.set_zones)

        # Face-zones toolbar above canvas <-> face_panel sync (loop-free)
        self.zones_toolbar_btn.toggled.connect(self._on_zones_toolbar_toggled)
        self.face_panel.zone_edit_toggled.connect(self._on_zone_edit_from_panel)
        self.zones_clear_btn.clicked.connect(self.face_panel.clear_zones)
        self.canvas.zones_changed.connect(
            lambda z: self._set_zone_count(len(z)))
        self.face_panel.zones_edited.connect(
            lambda z: self._set_zone_count(len(z)))

        # Blend (Смешивание) panel — variant-list based (Block G, task 4)
        self.blend_panel.selection_changed.connect(
            self._on_blend_variant_selection)
        self.blend_panel.blend_selected_requested.connect(
            self._on_blend_selected)
        self.blend_panel.preview_requested.connect(self._on_blend_preview)
        self.blend_panel.apply_requested.connect(self._on_blend_apply)
        self.blend_panel.auto_requested.connect(self._on_blend_auto)

        self.control_panel.set_ai_assistant_enabled(
            self.settings.get("use_llm_advisor", True))
        self.control_panel.ai_assistant_toggled.connect(self._on_ai_assistant_toggled)

        # Controller
        self.controller.progress_updated.connect(self._on_progress)
        self.controller.variants_ready.connect(self._on_variants_ready)
        self.controller.processing_complete.connect(self._on_complete)
        self.controller.processing_error.connect(self._on_error)
        self.controller.processing_cancelled.connect(self._on_cancelled)

    def _setup_status_bar(self):
        self.status_label = QLabel()
        self._last_status = ("status.ready", {})
        self.status_label.setText(tr("status.ready"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.gpu_label = QLabel()

        sb = self.statusBar()
        sb.addWidget(self.status_label)
        sb.addWidget(self.progress_bar)
        sb.addPermanentWidget(self.gpu_label)

        self._update_gpu_label()

        self.disk_label = QLabel()
        sb.addPermanentWidget(self.disk_label)
        self._update_disk_usage()

    def _update_gpu_label(self):
        device = self.settings.get("gpu_device", "auto")
        try:
            import torch
        except ImportError:
            # torch powers AI plugins but isn't required to launch the GUI
            # (traditional algorithms work without it).
            self.gpu_label.setText(tr("status.device_cpu_no_torch"))
            return
        if device == "cpu":
            self.gpu_label.setText(tr("status.device_cpu"))
        elif torch.cuda.is_available():
            idx = 0
            if device.startswith("cuda:"):
                try:
                    idx = int(device.split(":")[1])
                except (IndexError, ValueError):
                    idx = 0
            name = torch.cuda.get_device_name(idx)
            self.gpu_label.setText(tr("status.device_gpu", name=name))
        else:
            self.gpu_label.setText(tr("status.device_cpu_no_cuda"))

    def _update_disk_usage(self):
        usage_bytes = self.history_manager.disk_usage_bytes()
        usage_mb = usage_bytes / (1024 * 1024)
        usage_gb = usage_bytes / (1024 ** 3)
        if usage_gb >= 5.0:
            self.disk_label.setStyleSheet("color: red;")
            self.disk_label.setText(tr("status.disk_gb_warning", gb=usage_gb))
        elif usage_gb >= 1.0:
            self.disk_label.setText(tr("status.disk_gb", gb=usage_gb))
        else:
            self.disk_label.setText(tr("status.disk_mb", mb=usage_mb))

    # ─── Image loading ───

    @Slot()
    def _on_open_image(self):
        exts = " ".join(f"*{e}" for e in SUPPORTED_INPUT_FORMATS)
        path, _ = QFileDialog.getOpenFileName(
            self, tr("dialog.open_image_title"), "",
            tr("dialog.images_filter", exts=exts)
        )
        if path:
            self._load_image(Path(path))

    @Slot()
    def _on_open_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("dialog.open_dir_title"))
        if d:
            self.batch_mode_rb.setChecked(True)
            self.batch_panel.add_directory(d)

    def _load_image(self, path: Path):
        try:
            image, meta = read_image(path)
            self._current_image = image
            self._original_image = image.copy()
            self._current_meta = meta
            self._current_path = path
            self.canvas.set_before_image(image)
            self.canvas.set_after_image(image)
            self.canvas.fit_to_view()
            self._set_status("status.loaded", name=path.name,
                             w=meta['width'], h=meta['height'])
            self.history_panel.clear()
            self.history_manager.create_session()
            self.variant_store.clear()
            self.blend_panel.set_variants([])
        except Exception as e:
            QMessageBox.critical(self, tr("msg.error_title"),
                                 tr("msg.load_error_body", error=e))

    # ─── Auto config ───

    @Slot(bool)
    def _on_ai_assistant_toggled(self, enabled: bool):
        """Persist the assistant toggle so it survives restarts and matches Settings."""
        self.settings["use_llm_advisor"] = bool(enabled)
        try:
            save_settings(self.settings)
        except Exception:
            pass

    def _llm_gpu_layers(self) -> int:
        """How many LLM layers to offload to the GPU. Mirrors the image-device
        setting: 0 (CPU) when the user forced CPU, else -1 (offload all when the
        llama-cpp build supports CUDA; a CPU-only build ignores it)."""
        return 0 if self.settings.get("gpu_device") == "cpu" else -1

    def _run_llm_refine(self, analysis: dict, base_config: dict, on_done,
                        allow_deblur: bool = True, allow_icedit: bool = True,
                        allow_face: bool = True, blend_enabled: bool = False):
        """Refine config via the LLM advisor off the GUI thread, then call
        ``on_done(refined_config)`` on the GUI thread.

        If the advisor is disabled/unavailable, ``on_done`` is called
        synchronously with the unchanged config so callers have one code path.
        When *allow_deblur* is False the advisor is told not to emit SmartDeblur
        parameters. When *allow_icedit* is False the advisor is told not to emit
        ICEdit parameters. When *allow_face* is False the advisor is told not to
        emit CodeFormer face-restoration parameters. *blend_enabled* only makes
        the advisor aware that a later blend step may run; it never controls it.
        """
        if not self.control_panel.is_ai_assistant_enabled():
            on_done(base_config)
            return
        try:
            if self._llm_advisor is None:
                from upscaler.engine.llm_advisor import LLMAdvisor
                self._llm_advisor = LLMAdvisor(n_gpu_layers=self._llm_gpu_layers())
            if not self._llm_advisor.available():
                on_done(base_config)
                return
        except Exception:
            on_done(base_config)
            return

        from upscaler.ui.llm_worker import LLMRefineThread
        self._set_status("status.llm_refining")
        thread = LLMRefineThread(
            self._llm_advisor, self._current_image, analysis, base_config,
            allow_deblur=allow_deblur, allow_icedit=allow_icedit,
            allow_face=allow_face, blend_enabled=blend_enabled)
        # keep a reference so the thread is not garbage-collected mid-run
        self._refine_thread = thread

        def _finish(refined):
            on_done(refined)
            thread.quit()
            thread.wait()
            thread.deleteLater()
            self._refine_thread = None

        thread.done.connect(_finish)
        thread.start()

    @Slot()
    def _on_auto_config(self):
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        from upscaler.engine.analyzer import SourceAnalyzer
        from upscaler.engine.auto_config import AutoConfigurator

        analyzer = SourceAnalyzer()
        analysis = analyzer.analyze(self._current_image, self._current_meta)
        configurator = AutoConfigurator()

        scale = self.control_panel._get_selected_scale()
        enhance_only = self.control_panel.enhance_only_cb.isChecked()
        config = configurator.recommend(analysis, scale=scale, enhance_only=enhance_only,
                                         allow_predownscale=self.control_panel.is_predownscale_enabled())

        # SmartDeblur only participates in auto operations when its checkbox
        # (single tab) is ticked; otherwise strip it and forbid the advisor
        # from re-introducing it.
        allow_deblur = self.control_panel.is_deblur_enabled()
        if not allow_deblur:
            config.pop("deblur", None)
        allow_icedit = self.control_panel.is_icedit_enabled()
        if not allow_icedit:
            config.pop("icedit", None)
        allow_face = self.control_panel.is_face_enabled()
        if not allow_face:
            config.pop("face", None)
        if self.control_panel.is_blend_enabled():
            config["blend"] = {"enabled": True}
        else:
            config.pop("blend", None)

        def _apply(refined):
            self.control_panel.apply_auto_config(refined)
            self.deblur_panel.apply_config(refined.get("deblur") or {})
            self.icedit_panel.apply_config(refined.get("icedit") or {})
            self.face_panel.apply_config(refined.get("face") or {})
            self._set_status("status.auto_applied", desc=configurator.describe(refined))

        self._run_llm_refine(analysis, config, _apply,
                             allow_deblur=allow_deblur, allow_icedit=allow_icedit,
                             allow_face=allow_face,
                             blend_enabled=self.control_panel.is_blend_enabled())

    def _inject_face_zones(self, config: dict) -> dict:
        """Ручные зоны лиц действуют и в авто-конвейерах: если пользователь
        разметил зоны, а конфиг лица их не задал — добавить их."""
        zones = self.face_panel.get_zones()
        face_cfg = config.get("face")
        if zones and isinstance(face_cfg, dict) and face_cfg:
            face_cfg.setdefault("regions", zones)
        return config

    def _inject_blend(self, config: dict) -> dict:
        """Итерации доработки тоже смешивают, если чекбокс включён."""
        if self.control_panel.is_blend_enabled():
            config["blend"] = {"enabled": True}
        return config

    # --- Смешивание (вкладка) -------------------------------------------------
    #
    # Block G, task 4: источники — не история/строковые ключи, а варианты
    # (снимки шагов конвейера) из self.variant_store, наполняемого по
    # controller.variants_ready. Рецепт — пара {primary, secondary} + mode/
    # opacity (см. BlendPanel.get_recipe).

    def _reset_variants(self) -> None:
        """Сброс сессии вариантов (store + список на вкладке «Смешивание»).

        Вызывается в НАЧАЛЕ каждого прогона, способного породить варианты, чтобы
        вкладка отражала ТЕКУЩИЙ прогон, а не устаревшие варианты прошлого.
        `controller.variants_ready` эмитится только в ветке «result», поэтому
        прогон с ошибкой/отменой иначе оставил бы варианты прошлого прогона.
        """
        self.variant_store.clear()
        self.blend_panel.set_variants([])

    @Slot(str, list)
    def _on_variants_ready(self, job_id, variants):
        """Снимки шагов конвейера очередного прогона -> variant_store +
        список вариантов на вкладке «Смешивание»."""
        self.variant_store.clear()
        items = []
        for v in variants:
            try:
                img, _ = read_image(Path(v["path"]))
            except Exception:
                continue
            label = v.get("label", "")
            variant_id = self.variant_store.add(img, {"label": label})
            thumb = self.variant_store.get_thumbnail(variant_id)
            items.append((variant_id, thumb, label))
        self.blend_panel.set_variants(items)

    @Slot(int, int)
    def _on_blend_variant_selection(self, primary, secondary):
        """Показать выбранные варианты на холсте (по образцу истории)."""
        prim_img = self.variant_store.get_image(primary) if primary >= 0 else None
        sec_img = self.variant_store.get_image(secondary) if secondary >= 0 else None
        if prim_img is not None:
            self._current_image = prim_img
        before, after = compute_canvas_images(prim_img, sec_img)
        self.canvas.set_before_image(before)
        self.canvas.set_after_image(after)

    def _compose_blend_variants(self, recipe: dict):
        """recipe: {"primary","secondary","mode","opacity"} (id вариантов)."""
        base_img = self.variant_store.get_image(recipe["primary"])
        layer_img = self.variant_store.get_image(recipe["secondary"])
        if base_img is None or layer_img is None:
            QMessageBox.warning(self, tr("msg.blend_title"),
                                tr("msg.blend_sources_unavailable"))
            return None
        if base_img.shape[:2] != layer_img.shape[:2]:
            h, w = base_img.shape[:2]
            layer_img = cv2.resize(layer_img, (w, h),
                                   interpolation=cv2.INTER_LANCZOS4)
        return blend(base_img, layer_img, recipe["mode"], recipe["opacity"])

    @Slot(dict)
    def _on_blend_selected(self, recipe):
        """«Смешать выбранные»: результат идёт на холст и новым вариантом
        (для дальнейшего комбинирования), не в постоянную историю."""
        result = self._compose_blend_variants(recipe)
        if result is None:
            return
        self.canvas.set_after_image(result)
        self._current_image = result
        variant_id = self.variant_store.add(
            result, {"source": "blend_selected", "recipe": recipe})
        thumb = self.variant_store.get_thumbnail(variant_id)
        self.blend_panel.add_variant(variant_id, thumb, tr("blend.result_label"))
        self._set_status("status.blend_variant_added", id=variant_id)

    @Slot(dict)
    def _on_blend_preview(self, recipe):
        result = self._compose_blend_variants(recipe)
        if result is None:
            return
        self._last_blend_result = result
        self.canvas.set_after_image(result)
        self._set_status("status.blend_preview_ready")

    @Slot(dict)
    def _on_blend_apply(self, recipe):
        """«Применить»: результат сохраняется в постоянную историю."""
        result = self._compose_blend_variants(recipe)
        if result is None:
            return
        version = self.history_manager.add_version(
            result, {"source": "blend_manual", "recipe": recipe})
        self.history_panel.add_version(
            version, self.history_manager.get_thumbnail(version),
            {"source": "blend_manual"})
        self.canvas.set_after_image(result)
        self._current_image = result
        self._set_status("status.blend_saved", version=version)

    @Slot()
    def _on_blend_auto(self):
        """«Авто-подбор»: жадный поиск по вариантам, результат СРАЗУ
        показывается на холсте и добавляется вариантом (фикс бага: раньше
        рецепт только заполнял контролы, ничего не показывая)."""
        from upscaler.engine.blend_search import (
            make_proxy, greedy_blend_search, apply_recipe,
        )
        ids = self.variant_store.list_ids()
        images = {vid: self.variant_store.get_image(vid) for vid in ids}
        images = {k: v for k, v in images.items() if v is not None}
        if len(images) < 2:
            QMessageBox.information(
                self, tr("msg.blend_title"), tr("msg.blend_need_two_variants"))
            return
        base_key = next(iter(images))
        h, w = images[base_key].shape[:2]
        proxies = {}
        for key, img in images.items():
            p = img if img.shape[:2] == (h, w) else cv2.resize(
                img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            proxies[key] = make_proxy(p)
        recipe = greedy_blend_search(proxies)
        if recipe.get("base") is None:
            return
        result = apply_recipe(recipe, images)
        self.canvas.set_after_image(result)
        self._current_image = result
        variant_id = self.variant_store.add(
            result, {"source": "blend_auto", "recipe": recipe})
        thumb = self.variant_store.get_thumbnail(variant_id)
        self.blend_panel.add_variant(variant_id, thumb,
                                     tr("blend.auto_result_label"))
        n = len(recipe.get("layers", []))
        self._set_status("status.blend_auto_result", base=recipe.get('base'),
                         n=n, score=recipe.get('score', 0))

    # ─── Make beautiful (one-click) ───

    @Slot()
    def _on_make_beautiful(self):
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        if self.controller.is_running():
            return

        from upscaler.engine.analyzer import SourceAnalyzer
        from upscaler.engine.auto_config import AutoConfigurator

        self._set_status("status.analyzing")
        analyzer = SourceAnalyzer()
        analysis = analyzer.analyze(self._current_image, self._current_meta)
        configurator = AutoConfigurator()
        config = configurator.recommend(analysis, scale=4, enhance_only=False,
                                         allow_predownscale=self.control_panel.is_predownscale_enabled())

        # Honor the SmartDeblur checkbox: skip deblur unless the user enabled it.
        allow_deblur = self.control_panel.is_deblur_enabled()
        if not allow_deblur:
            config.pop("deblur", None)
        allow_icedit = self.control_panel.is_icedit_enabled()
        if not allow_icedit:
            config.pop("icedit", None)
        allow_face = self.control_panel.is_face_enabled()
        if not allow_face:
            config.pop("face", None)
        if self.control_panel.is_blend_enabled():
            config["blend"] = {"enabled": True}
        else:
            config.pop("blend", None)
        self.make_beautiful_btn.setEnabled(False)

        def _proceed(refined):
            self.control_panel.apply_auto_config(refined)
            self.deblur_panel.apply_config(refined.get("deblur") or {})
            self.icedit_panel.apply_config(refined.get("icedit") or {})
            self.face_panel.apply_config(refined.get("face") or {})
            desc = configurator.describe(refined)
            self._set_status("status.auto_processing", desc=desc)
            if self.controller.is_running():
                self.make_beautiful_btn.setEnabled(True)
                return
            tmp = Path(tempfile.mkdtemp()) / "input.png"
            write_image(self._current_image, tmp)
            refined["device"] = self.settings.get("gpu_device", "auto")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.process_btn.setEnabled(False)
            self._iter_state = {
                "active": True, "iteration": 1,
                "max": self.control_panel.get_max_refine_iterations(),
                "allow_deblur": allow_deblur, "allow_icedit": allow_icedit,
                "allow_face": allow_face,
            }
            refined = self._inject_face_zones(refined)
            refined = self._inject_blend(refined)
            self._reset_variants()
            self.controller.submit_job(str(tmp), refined)

        self._run_llm_refine(analysis, config, _proceed,
                             allow_deblur=allow_deblur, allow_icedit=allow_icedit,
                             allow_face=allow_face,
                             blend_enabled=self.control_panel.is_blend_enabled())

    # ─── Processing ───

    @Slot()
    def _on_process(self):
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        if self.controller.is_running():
            return

        tmp = Path(tempfile.mkdtemp()) / "input.png"
        write_image(self._current_image, tmp)

        config = self.control_panel.get_pipeline_config()
        config["device"] = self.settings.get("gpu_device", "auto")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.make_beautiful_btn.setEnabled(False)
        self._iter_state = {
            "active": True, "iteration": 1,
            "max": self.control_panel.get_max_refine_iterations(),
            "allow_deblur": self.control_panel.is_deblur_enabled(),
            "allow_icedit": self.control_panel.is_icedit_enabled(),
            "allow_face": self.control_panel.is_face_enabled(),
        }
        self._reset_variants()
        self.controller.submit_job(str(tmp), config)

    def _on_deblur_preview(self, deblur_cfg: dict):
        """Run a deblur-only pass on the current image for preview."""
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        if self.controller.is_running():
            return

        tmp = Path(tempfile.mkdtemp()) / "input.png"
        write_image(self._current_image, tmp)

        # Deblur only: enhance-only with no upscaler runs just the pre-process
        # stages (deblur) at native resolution.
        config = {
            "scale": 2,
            "enhance_only": True,
            "deblur": deblur_cfg,
            "denoise": {},
            "adjust": {},
            "upscale": {},
            "post": {},
            "device": self.settings.get("gpu_device", "auto"),
        }
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.make_beautiful_btn.setEnabled(False)
        self._reset_variants()
        self.controller.submit_job(str(tmp), config)

    def _on_icedit_preview(self, icedit_cfg: dict):
        """Run an ICEdit-only pass on the current image for preview."""
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        if self.controller.is_running():
            return
        if not (icedit_cfg.get("instruction") or "").strip():
            QMessageBox.warning(self, tr("msg.no_instruction_title"),
                                tr("msg.no_instruction_body"))
            return

        tmp = Path(tempfile.mkdtemp()) / "input.png"
        write_image(self._current_image, tmp)

        config = {
            "scale": 2,
            "enhance_only": True,
            "icedit": icedit_cfg,
            "denoise": {},
            "adjust": {},
            "upscale": {},
            "post": {},
            "device": self.settings.get("gpu_device", "auto"),
        }
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.make_beautiful_btn.setEnabled(False)
        self._reset_variants()
        self.controller.submit_job(str(tmp), config)

    def _on_face_preview(self, face_cfg: dict):
        """Run a face-restoration-only pass on the current image for preview."""
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        if self.controller.is_running():
            return

        tmp = Path(tempfile.mkdtemp()) / "input.png"
        write_image(self._current_image, tmp)

        config = {
            "scale": 2,
            "enhance_only": True,
            "face": face_cfg,
            "denoise": {},
            "adjust": {},
            "upscale": {},
            "post": {},
            "device": self.settings.get("gpu_device", "auto"),
        }
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.make_beautiful_btn.setEnabled(False)
        self._reset_variants()
        self.controller.submit_job(str(tmp), config)

    @Slot(bool)
    def _on_zones_toolbar_toggled(self, checked: bool):
        # Кнопка панели «Лица» — источник истины; guard от петли.
        if self.face_panel.zone_edit_btn.isChecked() != checked:
            self.face_panel.zone_edit_btn.setChecked(checked)

    @Slot(bool)
    def _on_zone_edit_from_panel(self, checked: bool):
        self.canvas.set_zone_edit_mode(checked)
        if self.zones_toolbar_btn.isChecked() != checked:
            self.zones_toolbar_btn.setChecked(checked)

    @Slot()
    def _on_cancel(self):
        if self.controller._current_job_id:
            self.controller.cancel_job(self.controller._current_job_id)

    @Slot(str, str, int, str)
    def _on_progress(self, job_id, stage, percent, message):
        self.progress_bar.setValue(percent)
        self._set_status("status.progress", stage=stage, message=message)

    @Slot(str, str, dict)
    def _on_complete(self, job_id, output_path, metrics):
        # During an active refinement loop keep controls disabled until finalize.
        if not (self._iter_state and self._iter_state.get("active")):
            self.progress_bar.setVisible(False)
            self.process_btn.setEnabled(True)
            self.make_beautiful_btn.setEnabled(True)

        try:
            result_image, _ = read_image(Path(output_path))

            # Check if this is a batch job
            if self._batch_queue:
                out_dir = Path(self._batch_output_dir) if self._batch_output_dir else self._batch_current_source.parent
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{self._batch_current_source.stem}_upscaled.png"
                write_image(result_image, out_path)
                self._batch_index += 1
                self._process_next_batch()
                return

            self.canvas.set_before_image(self._current_image)
            self.canvas.set_after_image(result_image)
            self._current_image = result_image

            bit_depth = self._current_meta.get("bit_depth", 8)
            version = self.history_manager.add_version(result_image, metrics, bit_depth)
            self.history_panel.add_version(version, self.history_manager.get_thumbnail(version), metrics)
            self._set_status("status.complete_metrics",
                             brisque=metrics.get('brisque', 0), niqe=metrics.get('niqe', 0))
            self._update_disk_usage()

            if self._iter_state and self._iter_state.get("active"):
                self._maybe_refine(result_image)
                return
        except Exception as e:
            QMessageBox.critical(self, tr("msg.error_title"),
                                 tr("msg.result_load_error_body", error=e))

    def _maybe_refine(self, result_image):
        """Evaluate the latest result and, if needed, queue another pass."""
        from upscaler.engine.llm_advisor import should_continue_refinement
        st = self._iter_state
        if (st["iteration"] >= st["max"]
                or not self.control_panel.is_ai_assistant_enabled()):
            self._finalize_iteration()
            return
        try:
            if self._llm_advisor is None:
                from upscaler.engine.llm_advisor import LLMAdvisor
                self._llm_advisor = LLMAdvisor(n_gpu_layers=self._llm_gpu_layers())
            if not self._llm_advisor.available():
                self._finalize_iteration()
                return
        except Exception:
            self._finalize_iteration()
            return

        from upscaler.engine.analyzer import SourceAnalyzer
        analysis = SourceAnalyzer().analyze(result_image, self._current_meta,
                                            detect_faces=False)
        self._set_status("status.evaluating", iter=st['iteration'], max=st['max'])

        from upscaler.ui.llm_worker import LLMEvaluateThread
        from upscaler.engine.llm_advisor import should_continue_refinement
        thread = LLMEvaluateThread(
            self._llm_advisor, result_image, analysis,
            allow_deblur=st["allow_deblur"], allow_icedit=st["allow_icedit"],
            allow_face=st.get("allow_face", True))
        self._eval_thread = thread

        def _on_eval(verdict):
            thread.quit()
            thread.wait()
            thread.deleteLater()
            self._eval_thread = None
            cont = should_continue_refinement(
                st["iteration"], st["max"], verdict.get("satisfied", True))
            cfg = verdict.get("config")
            if cont and cfg:
                st["iteration"] += 1
                cfg["device"] = self.settings.get("gpu_device", "auto")
                cfg = self._inject_face_zones(cfg)
                cfg = self._inject_blend(cfg)
                tmp = Path(tempfile.mkdtemp()) / "input.png"
                write_image(result_image, tmp)  # cumulative: input = result
                self._set_status("status.refining", iter=st['iteration'], max=st['max'])
                self.controller.submit_job(str(tmp), cfg)
            else:
                self._finalize_iteration()

        thread.done.connect(_on_eval)
        thread.start()

    def _finalize_iteration(self, metrics_text: str | None = None):
        """End the refinement loop and restore the UI."""
        self._iter_state = None
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.make_beautiful_btn.setEnabled(True)
        if metrics_text is None:
            self._set_status("status.done")
        else:
            self.status_label.setText(metrics_text)

    @Slot(str, str, str, bool)
    def _on_error(self, job_id, stage, error, recoverable):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.make_beautiful_btn.setEnabled(True)
        self._set_status("status.error_stage", stage=stage)
        QMessageBox.critical(self, tr("msg.processing_error_title"),
                             tr("msg.processing_error_body", stage=stage, error=error))
        self._iter_state = None

        # If batch, skip to next
        if self._batch_queue:
            log.warning(f"Batch: error on {self._batch_current_source}: {error}")
            self._batch_index += 1
            self._process_next_batch()

    @Slot(str)
    def _on_cancelled(self, job_id):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.make_beautiful_btn.setEnabled(True)
        self._set_status("status.cancelled")
        self._batch_queue = []
        self._iter_state = None

    # ─── Revert & Compare ───

    @Slot()
    @Slot(int)
    def _on_revert_to_original(self, version=None):
        """Revert to original unprocessed image."""
        if self._original_image is not None:
            self._current_image = self._original_image.copy()
            self.canvas.set_before_image(self._original_image)
            self.canvas.set_after_image(self._original_image)
            self._set_status("status.reverted")

    @Slot(int)
    def _on_compare(self, version):
        """Open comparison dialog: original vs current."""
        if self._original_image is None or self._current_image is None:
            return
        from upscaler.ui.compare_dialog import CompareDialog
        dlg = CompareDialog(self._original_image, self._current_image, self)
        dlg.exec()

    @Slot(int, int)
    def _on_history_selection(self, primary, secondary):
        """Show selected history versions on the canvas (display/compare) and
        make the primary the working image for Save and the next pass."""
        prim_img = (self.history_manager.get_version_image(primary)
                    if primary >= 0 else None)
        sec_img = (self.history_manager.get_version_image(secondary)
                   if secondary >= 0 else None)
        if prim_img is not None:
            self._current_image = prim_img
        before, after = compute_canvas_images(prim_img, sec_img)
        self.canvas.set_before_image(before)
        self.canvas.set_after_image(after)

    @Slot(int)
    def _on_history_delete(self, version):
        """Delete a version from disk when removed from the history panel."""
        self.history_manager.delete_version(version)
        self._update_disk_usage()

    # ─── Batch processing ───

    @Slot()
    def _on_batch_process(self):
        files = self.batch_panel.get_files()
        if not files:
            QMessageBox.warning(self, tr("msg.no_files_title"), tr("msg.no_files_body"))
            return
        self._batch_queue = list(files)
        self._batch_mode = "manual"
        self._batch_output_dir = self.batch_panel.get_output_dir()
        self._batch_index = 0
        self._reset_variants()
        self._process_next_batch()

    @Slot()
    def _on_batch_beautiful(self):
        files = self.batch_panel.get_files()
        if not files:
            QMessageBox.warning(self, tr("msg.no_files_title"), tr("msg.no_files_body"))
            return
        self._batch_queue = list(files)
        self._batch_mode = "beautiful"
        self._batch_output_dir = self.batch_panel.get_output_dir()
        self._batch_index = 0
        self._reset_variants()
        self._process_next_batch()

    def _process_next_batch(self):
        if self._batch_index >= len(self._batch_queue):
            self.batch_panel.batch_progress.setText(
                tr("status.batch_done", n=len(self._batch_queue))
            )
            self.batch_panel.progress_bar.setVisible(False)
            self._batch_queue = []
            self._set_status("status.batch_complete")
            return

        path = self._batch_queue[self._batch_index]
        self.batch_panel.set_progress(
            self._batch_index + 1, len(self._batch_queue), path.name
        )

        try:
            image, meta = read_image(path)
        except Exception as e:
            log.error(f"Batch: не удалось прочитать {path}: {e}")
            self._batch_index += 1
            self._process_next_batch()
            return

        tmp = Path(tempfile.mkdtemp()) / "input.png"
        write_image(image, tmp)

        if self._batch_mode == "beautiful":
            from upscaler.engine.analyzer import SourceAnalyzer
            from upscaler.engine.auto_config import AutoConfigurator
            analyzer = SourceAnalyzer()
            analysis = analyzer.analyze(image, meta)
            configurator = AutoConfigurator()
            config = configurator.recommend(analysis,
                                             allow_predownscale=self.control_panel.is_predownscale_enabled())
        else:
            config = self.control_panel.get_pipeline_config()

        config["device"] = self.settings.get("gpu_device", "auto")
        self._batch_current_source = path
        self.process_btn.setEnabled(False)
        self.make_beautiful_btn.setEnabled(False)
        self.controller.submit_job(str(tmp), config)

    # ─── Colorization ───

    @Slot(str, dict)
    def _on_colorize_photo(self, model_name, params):
        """Colorize current photo using selected model."""
        if self._current_image is None:
            QMessageBox.warning(self, tr("msg.no_image_title"), tr("msg.no_image_body"))
            return
        if self.controller.is_running():
            return

        tmp = Path(tempfile.mkdtemp()) / "input.png"
        write_image(self._current_image, tmp)

        config = {
            "scale": 1,
            "enhance_only": True,
            "denoise": {},
            "adjust": {},
            "upscale": {},
            "post": {},
            "colorize": {model_name: params},
            "device": self.settings.get("gpu_device", "auto"),
        }
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self._set_status("status.colorizing_photo", model=model_name)
        self.controller.submit_job(str(tmp), config)

    @Slot(str, dict, str, str)
    def _on_colorize_video(self, model_name, params, input_path, output_path):
        """Colorize video frame by frame."""
        self._set_status("status.colorizing_video", model=model_name)

        import threading
        thread = threading.Thread(
            target=self._colorize_video_worker,
            args=(model_name, params, input_path, output_path),
            daemon=True,
        )
        thread.start()

    def _colorize_video_worker(self, model_name, params, input_path, output_path):
        """Worker thread for video colorization."""
        try:
            from upscaler.plugins.registry import PluginRegistry
            registry = PluginRegistry()
            registry.discover_builtin()
            plugin_cls = registry.get(model_name)
            if not plugin_cls:
                return

            device = self.settings.get("gpu_device", "auto")
            plugin = plugin_cls()
            plugin.initialize(device)

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            import cv2 as cv2_mod
            fourcc = cv2_mod.VideoWriter_fourcc(*'mp4v')
            writer = cv2_mod.VideoWriter(output_path, fourcc, fps, (w, h))

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2_mod.cvtColor(frame, cv2_mod.COLOR_BGR2RGB)

                if hasattr(plugin, 'process_video_frame'):
                    result = plugin.process_video_frame(frame_rgb, params)
                else:
                    result = plugin.process(frame_rgb, params)

                result_bgr = cv2_mod.cvtColor(result, cv2_mod.COLOR_RGB2BGR)
                writer.write(result_bgr)

                frame_idx += 1
                # Update UI from thread (safe via signal would be better, but simple approach)
                if frame_idx % 5 == 0:
                    from PySide6.QtCore import QMetaObject, Q_ARG
                    # Simple approach: just update the panel
                    try:
                        self.colorize_panel.set_video_progress(frame_idx, total_frames)
                    except Exception:
                        pass

            cap.release()
            writer.release()
            plugin.cleanup()
            self.colorize_panel.video_progress.setText(
                tr("status.video_done", path=output_path))
            self.colorize_panel.video_progress_bar.setVisible(False)
        except Exception as e:
            log.error(f"Video colorization failed: {e}", exc_info=True)
            self.colorize_panel.video_progress.setText(tr("status.video_error", error=e))

    # ─── Save / Settings / History ───

    @Slot()
    def _on_save_result(self):
        if self._current_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, tr("dialog.save_result_title"), "",
            tr("dialog.save_filter")
        )
        if path:
            write_image(self._current_image, Path(path))
            self._set_status("status.saved", path=path)

    def _on_max_iter_changed(self, value: int):
        """Persist the Single-panel max-refine-iterations value to settings."""
        self.settings["max_refine_iterations"] = int(value)
        save_settings(self.settings)

    @Slot(bool)
    def _on_predownscale_toggled(self, enabled: bool):
        self.settings["auto_predownscale"] = bool(enabled)
        save_settings(self.settings)

    @Slot(str, bool)
    def _on_section_toggled(self, key: str, expanded: bool):
        sections = dict(self.settings.get("panel_sections") or {})
        sections[key] = bool(expanded)
        self.settings["panel_sections"] = sections
        save_settings(self.settings)

    @Slot()
    def _on_settings(self):
        from upscaler.ui.settings_dialog import SettingsDialog
        old_device = self.settings.get("gpu_device", "auto")
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings.update(dlg.get_settings())
            save_settings(self.settings)
            self._update_gpu_label()
            if self.settings.get("gpu_device") != old_device:
                self.controller.stop_engine()
                self.controller.start_engine()

    @Slot()
    def _on_clear_history(self):
        reply = QMessageBox.question(
            self, tr("msg.clear_history_title"),
            tr("msg.clear_history_body")
        )
        if reply == QMessageBox.StandardButton.Yes:
            import shutil
            shutil.rmtree(self.history_manager.base_dir, ignore_errors=True)
            self.history_manager.base_dir.mkdir(parents=True, exist_ok=True)
            self.history_manager.create_session()
            self.history_panel.clear()
            self.variant_store.clear()
            self.blend_panel.set_variants([])

    def closeEvent(self, event):
        self.settings["window_geometry"] = bytes(
            self.saveGeometry().toBase64()).decode("ascii")
        save_settings(self.settings)
        self.controller.stop_engine()
        super().closeEvent(event)
