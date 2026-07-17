"""Left panel: presets, scale, plugin checkboxes, sliders."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QComboBox, QRadioButton,
    QCheckBox, QSlider, QLabel, QGroupBox, QButtonGroup, QProgressBar,
    QScrollArea, QHBoxLayout, QSpinBox, QListWidget, QListWidgetItem,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal

from upscaler.presets.loader import PresetLoader
from upscaler.plugins.registry import PluginRegistry
from upscaler.engine.pipeline import (
    PipelineExecutor, resolve_step_order, step_label,
)
from upscaler.ui.collapsible import CollapsibleSection
from upscaler.ui.i18n import tr


# Разбиение корректоров («adjuster») по подсекциям вкладки «Коррекция».
# Любой adjuster, не перечисленный ниже, попадает в «Детали» (fallback).
_ADJUSTER_SUBSECTIONS = {
    "adjust_tone": ("section.adjust_tone",
                    ["Auto Levels", "Auto Tone", "Auto Contrast",
                     "Brightness", "Contrast", "Shadows/Highlights",
                     "Dodge & Burn"]),
    "adjust_color": ("section.adjust_color",
                     ["White Balance", "Auto Color", "Saturation",
                      "Vibrance", "Split Toning"]),
    "adjust_detail": ("section.adjust_detail",
                      ["Clarity", "Dehaze", "Sharpness",
                       "Refocus", "Skin Smooth", "Optics"]),
}

# Заголовки секций верхнего уровня: ключ секции -> i18n-ключ. Используется и
# при первичном создании, и в retranslate().
_SECTION_TITLE_KEYS = {
    "preset_scale": "section.preset_scale",
    "upscaler": "section.upscaler",
    "denoise": "section.denoise",
    "adjust": "section.adjust",
    "adjust_tone": "section.adjust_tone",
    "adjust_color": "section.adjust_color",
    "adjust_detail": "section.adjust_detail",
    "restore": "section.restore",
    "edit": "section.edit",
    "blend": "section.blend",
    "ai": "section.ai",
    "order": "section.order",
}


class ControlPanel(QWidget):
    # Signals emitted for MainWindow to connect to its own buttons
    auto_config_requested = Signal()
    max_iter_changed = Signal(int)
    ai_assistant_toggled = Signal(bool)
    variants_toggled = Signal(bool)
    predownscale_toggled = Signal(bool)
    section_toggled = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._preset_loader = PresetLoader()
        self._registry = PluginRegistry()
        self._registry.discover_builtin()
        self._checkboxes: dict[str, QCheckBox] = {}
        self._sliders: dict[str, QSlider] = {}
        self._combos: dict[str, QComboBox] = {}
        self._deblur_override: dict = {}
        self._icedit_override: dict = {}
        self._face_override: dict = {}
        self._predownscale_override: dict = {}
        self._order_override: list = []  # LLM/auto-chosen step sequence, if any
        self._user_order: list = []  # порядок, заданный пользователем вручную
        self._sections: dict = {}
        self._param_boxes: dict = {}
        self._setup_ui()

    def _add_section(self, layout, key: str, title: str,
                     expanded: bool) -> CollapsibleSection:
        section = CollapsibleSection(title, expanded=expanded)
        section.toggled.connect(
            lambda v, k=key: self.section_toggled.emit(k, v))
        self._sections[key] = section
        layout.addWidget(section)
        return section

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Пресет и масштаб ------------------------------------------------
        preset_scale = self._add_section(layout, "preset_scale",
                                         tr("section.preset_scale"), True)
        self.preset_label = QLabel(tr("panel.preset_label"))
        preset_scale.add_widget(self.preset_label)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem(tr("panel.preset_manual"))
        for preset in self._preset_loader.list_presets():
            self.preset_combo.addItem(preset["name"])
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_scale.add_widget(self.preset_combo)

        self.scale_box = QGroupBox(tr("panel.scale_group_title"))
        scale_layout = QVBoxLayout()
        self.scale_group = QButtonGroup()
        for s in [2, 4, 8, 16]:
            rb = QRadioButton(f"{s}x")
            rb.setProperty("scale", s)
            self.scale_group.addButton(rb)
            scale_layout.addWidget(rb)
            if s == 4:
                rb.setChecked(True)
        self.enhance_only_cb = QCheckBox(tr("panel.enhance_only"))
        scale_layout.addWidget(self.enhance_only_cb)
        self.predownscale_cb = QCheckBox(tr("panel.predownscale"))
        self.predownscale_cb.setChecked(True)
        self.predownscale_cb.setToolTip(tr("panel.predownscale_tooltip"))
        self.predownscale_cb.toggled.connect(self.predownscale_toggled.emit)
        scale_layout.addWidget(self.predownscale_cb)
        self.scale_box.setLayout(scale_layout)
        preset_scale.add_widget(self.scale_box)

        # --- Апскейлер (один активный — комбобокс вместо чекбоксов) ----------
        upscaler_section = self._add_section(layout, "upscaler",
                                             tr("section.upscaler"), True)
        self.upscaler_combo = QComboBox()
        self.upscaler_combo.addItem(tr("panel.upscaler_none"), None)
        for plugin in self._registry.list_plugins("upscaler"):
            self.upscaler_combo.addItem(plugin.name, plugin.name)
        default_idx = self.upscaler_combo.findData("Real-ESRGAN")
        self.upscaler_combo.setCurrentIndex(
            default_idx if default_idx >= 0 else
            (1 if self.upscaler_combo.count() > 1 else 0))
        upscaler_section.add_widget(self.upscaler_combo)

        # --- Шумоподавление ----------------------------------------------------
        denoise_section = self._add_section(layout, "denoise",
                                            tr("section.denoise"), False)
        for plugin in self._registry.list_plugins("denoiser"):
            cb = QCheckBox(plugin.name)
            self._checkboxes[plugin.name] = cb
            denoise_section.add_widget(cb)
            self._add_param_controls(denoise_section.body_layout, plugin)

        # --- Коррекция (подсекции: Тон / Цвет / Детали) -----------------------
        adjust_section = self._add_section(layout, "adjust",
                                           tr("section.adjust"), False)
        adjusters_by_name = {p.name: p
                             for p in self._registry.list_plugins("adjuster")}
        assigned = set()
        for key, (title_key, names) in _ADJUSTER_SUBSECTIONS.items():
            sub = self._add_section(adjust_section.body_layout, key,
                                    tr(title_key), True)
            for name in names:
                plugin = adjusters_by_name.get(name)
                if plugin is None:
                    continue
                assigned.add(name)
                cb = QCheckBox(plugin.name)
                self._checkboxes[plugin.name] = cb
                sub.add_widget(cb)
                self._add_param_controls(sub.body_layout, plugin)
        # Любой adjuster вне явных списков — fallback в «Детали».
        detail_section = self._sections["adjust_detail"]
        for name, plugin in adjusters_by_name.items():
            if name in assigned:
                continue
            cb = QCheckBox(plugin.name)
            self._checkboxes[plugin.name] = cb
            detail_section.add_widget(cb)
            self._add_param_controls(detail_section.body_layout, plugin)

        # --- Восстановление (deblur / лица) -----------------------------------
        restore_section = self._add_section(layout, "restore",
                                            tr("section.restore"), True)
        for plugin in self._registry.list_plugins("deblur"):
            cb = QCheckBox(plugin.name)
            self._checkboxes[plugin.name] = cb
            restore_section.add_widget(cb)
        face_cb = QCheckBox(tr("panel.face_restore"))
        face_cb.setChecked(True)
        self._checkboxes["CodeFormer"] = face_cb
        restore_section.add_widget(face_cb)
        self.restore_hint_label = QLabel(tr("panel.restore_hint"))
        self.restore_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.restore_hint_label.setWordWrap(True)
        restore_section.add_widget(self.restore_hint_label)

        # --- Редактирование (ICEdit, instruction-based) ------------------------
        edit_section = self._add_section(layout, "edit",
                                         tr("section.edit"), False)
        icedit_cb = QCheckBox("ICEdit")
        self._checkboxes["ICEdit"] = icedit_cb
        edit_section.add_widget(icedit_cb)
        self.edit_hint_label = QLabel(tr("panel.edit_hint"))
        self.edit_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.edit_hint_label.setWordWrap(True)
        edit_section.add_widget(self.edit_hint_label)

        # --- Смешивание (авто-смешивание промежуточных вариантов) --------------
        blend_section = self._add_section(layout, "blend",
                                          tr("section.blend"), False)
        self.blend_cb = QCheckBox(tr("panel.blend_checkbox"))
        self.blend_cb.setChecked(False)
        blend_section.add_widget(self.blend_cb)
        self.blend_hint_label = QLabel(tr("panel.blend_hint"))
        self.blend_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.blend_hint_label.setWordWrap(True)
        blend_section.add_widget(self.blend_hint_label)

        # --- ИИ-ассистент и доработка -------------------------------------------
        ai_section = self._add_section(layout, "ai",
                                       tr("section.ai"), False)
        self.ai_assistant_cb = QCheckBox(tr("panel.ai_assistant_checkbox"))
        self.ai_assistant_cb.setChecked(True)
        self.ai_assistant_cb.toggled.connect(self.ai_assistant_toggled.emit)
        ai_section.add_widget(self.ai_assistant_cb)
        self.ai_hint_label = QLabel(tr("panel.ai_hint"))
        self.ai_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.ai_hint_label.setWordWrap(True)
        ai_section.add_widget(self.ai_hint_label)

        self.variants_cb = QCheckBox(tr("panel.variants_checkbox"))
        self.variants_cb.setChecked(False)
        self.variants_cb.toggled.connect(self.variants_toggled.emit)
        ai_section.add_widget(self.variants_cb)
        self.variants_hint_label = QLabel(tr("panel.variants_hint"))
        self.variants_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.variants_hint_label.setWordWrap(True)
        ai_section.add_widget(self.variants_hint_label)

        refine_row = QWidget()
        refine_layout = QHBoxLayout(refine_row)
        refine_layout.setContentsMargins(0, 0, 0, 0)
        self.max_iter_label = QLabel(tr("panel.max_iter_label"))
        refine_layout.addWidget(self.max_iter_label)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1, 10)
        self.max_iter_spin.setValue(3)
        self.max_iter_spin.setToolTip(tr("panel.max_iter_tooltip"))
        self.max_iter_spin.valueChanged.connect(self.max_iter_changed.emit)
        refine_layout.addWidget(self.max_iter_spin)
        ai_section.add_widget(refine_row)

        # --- Порядок обработки (drag-and-drop список шагов конвейера) ----------
        order_section = self._add_section(layout, "order",
                                          tr("section.order"), False)
        self.order_list = QListWidget()
        self.order_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self.order_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.order_list.setFixedHeight(220)
        self._populate_order_list(list(PipelineExecutor.DEFAULT_ORDER))
        self.order_list.model().rowsMoved.connect(self._on_order_moved)
        order_section.add_widget(self.order_list)
        order_btns_row = QWidget()
        order_btns = QHBoxLayout(order_btns_row)
        order_btns.setContentsMargins(0, 0, 0, 0)
        self.order_up_btn = QPushButton(tr("panel.order_up"))
        self.order_down_btn = QPushButton(tr("panel.order_down"))
        self.order_reset_btn = QPushButton(tr("panel.order_reset"))
        self.order_up_btn.clicked.connect(lambda: self._move_order_item(-1))
        self.order_down_btn.clicked.connect(lambda: self._move_order_item(1))
        self.order_reset_btn.clicked.connect(self._reset_order)
        order_btns.addWidget(self.order_up_btn)
        order_btns.addWidget(self.order_down_btn)
        order_btns.addWidget(self.order_reset_btn)
        order_section.add_widget(order_btns_row)
        self.order_hint_label = QLabel(tr("panel.order_hint"))
        self.order_hint_label.setStyleSheet("color: gray; font-size: 11px;")
        self.order_hint_label.setWordWrap(True)
        order_section.add_widget(self.order_hint_label)

        # Progress (shown during processing) — вне секций
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _add_param_controls(self, layout, plugin):
        """Build slider/combo rows for a plugin's params_schema.

        Rows are collected into a container box, hidden until the plugin's
        checkbox is enabled (see `_param_boxes` / `section_toggled`).
        """
        if not plugin.params_schema:
            return
        box = QWidget()
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(16, 0, 0, 0)
        box_layout.setSpacing(2)
        for pname, schema in plugin.params_schema.items():
            if schema.get("ui") == "combo":
                row = QHBoxLayout()
                row.addWidget(QLabel(f"  {pname}:"))
                combo = QComboBox()
                labels = schema.get("labels", {})
                for opt in schema.get("options", []):
                    combo.addItem(labels.get(opt, opt), opt)
                self._combos[f"{plugin.name}.{pname}"] = combo
                row.addWidget(combo)
                box_layout.addLayout(row)
            elif schema.get("ui") == "slider" and schema.get("type") == "number":
                row = QHBoxLayout()
                row.addWidget(QLabel(f"  {pname}:"))
                slider = QSlider(Qt.Orientation.Horizontal)
                mn = int(schema.get("minimum", 0) * 100)
                mx = int(schema.get("maximum", 1) * 100)
                df = int(schema.get("default", 0.5) * 100)
                slider.setRange(mn, mx)
                slider.setValue(df)
                self._sliders[f"{plugin.name}.{pname}"] = slider
                row.addWidget(slider)
                box_layout.addLayout(row)
            elif schema.get("ui") == "slider" and schema.get("type") == "integer":
                row = QHBoxLayout()
                row.addWidget(QLabel(f"  {pname}:"))
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(int(schema.get("minimum", 0)),
                                int(schema.get("maximum", 100)))
                slider.setValue(int(schema.get("default", 0)))
                self._sliders[f"{plugin.name}.{pname}"] = slider
                row.addWidget(slider)
                box_layout.addLayout(row)
        self._param_boxes[plugin.name] = box
        box.setVisible(False)
        cb = self._checkboxes.get(plugin.name)
        if cb is not None:
            cb.toggled.connect(box.setVisible)
        layout.addWidget(box)

    def retranslate(self) -> None:
        """Переустановить все видимые тексты панели из текущего языка (tr())."""
        for key, i18n_key in _SECTION_TITLE_KEYS.items():
            section = self._sections.get(key)
            if section is not None:
                section.header_btn.setText(tr(i18n_key))

        self.preset_label.setText(tr("panel.preset_label"))
        # Пункт «(Вручную)» всегда индекс 0 — просто обновляем его подпись;
        # его выбор не зависит от текста (детекция по currentIndex()==0).
        self.preset_combo.setItemText(0, tr("panel.preset_manual"))

        self.scale_box.setTitle(tr("panel.scale_group_title"))
        self.enhance_only_cb.setText(tr("panel.enhance_only"))
        self.predownscale_cb.setText(tr("panel.predownscale"))
        self.predownscale_cb.setToolTip(tr("panel.predownscale_tooltip"))

        self.upscaler_combo.setItemText(0, tr("panel.upscaler_none"))

        face_cb = self._checkboxes.get("CodeFormer")
        if face_cb is not None:
            face_cb.setText(tr("panel.face_restore"))
        self.restore_hint_label.setText(tr("panel.restore_hint"))

        self.edit_hint_label.setText(tr("panel.edit_hint"))

        self.blend_cb.setText(tr("panel.blend_checkbox"))
        self.blend_hint_label.setText(tr("panel.blend_hint"))

        self.ai_assistant_cb.setText(tr("panel.ai_assistant_checkbox"))
        self.ai_hint_label.setText(tr("panel.ai_hint"))
        self.variants_cb.setText(tr("panel.variants_checkbox"))
        self.variants_hint_label.setText(tr("panel.variants_hint"))
        self.max_iter_label.setText(tr("panel.max_iter_label"))
        self.max_iter_spin.setToolTip(tr("panel.max_iter_tooltip"))

        self.order_up_btn.setText(tr("panel.order_up"))
        self.order_down_btn.setText(tr("panel.order_down"))
        self.order_reset_btn.setText(tr("panel.order_reset"))
        self.order_hint_label.setText(tr("panel.order_hint"))
        # Подписи шагов в списке порядка тоже зависят от языка (step_label).
        self._populate_order_list(self.get_order())

    def get_section_states(self) -> dict:
        return {k: s.is_expanded() for k, s in self._sections.items()}

    def set_section_states(self, states: dict) -> None:
        for key, expanded in (states or {}).items():
            section = self._sections.get(key)
            if section is not None:
                section.set_expanded(bool(expanded))

    def get_pipeline_config(self) -> dict:
        """Build pipeline config dict from current UI state."""
        config = {
            "scale": self._get_selected_scale(),
            "enhance_only": self.enhance_only_cb.isChecked(),
            "denoise": {},
            "adjust": {},
            "upscale": {},
            "post": {},
            "deblur": {},
            "icedit": {},
            "face": {},
        }

        plugin_name = self.upscaler_combo.currentData()
        if plugin_name:
            config["upscale"] = {"plugin": plugin_name,
                                 "scale": config["scale"]}

        for plugin in self._registry.list_plugins("denoiser"):
            cb = self._checkboxes.get(plugin.name)
            if cb and cb.isChecked():
                params = self._get_plugin_params(plugin.name, plugin.params_schema)
                config["denoise"][plugin.name] = params

        for plugin in self._registry.list_plugins("adjuster"):
            cb = self._checkboxes.get(plugin.name)
            if cb and cb.isChecked():
                params = self._get_plugin_params(plugin.name, plugin.params_schema)
                config["adjust"][plugin.name] = params

        for plugin in self._registry.list_plugins("deblur"):
            cb = self._checkboxes.get(plugin.name)
            if cb and cb.isChecked():
                config["deblur"] = (dict(self._deblur_override)
                                    if self._deblur_override else {"auto": True})
                break

        ic_cb = self._checkboxes.get("ICEdit")
        if ic_cb and ic_cb.isChecked() and self._icedit_override:
            config["icedit"] = dict(self._icedit_override)

        face_cb = self._checkboxes.get("CodeFormer")
        if face_cb and face_cb.isChecked():
            config["face"] = (dict(self._face_override) if self._face_override
                              else {"enabled": True, "fidelity": 0.7,
                                    "upscale_background": False})

        if self.blend_cb.isChecked():
            config["blend"] = {"enabled": True}

        config["predownscale_enabled"] = self.predownscale_cb.isChecked()
        if self._predownscale_override and config["predownscale_enabled"]:
            config["predownscale"] = dict(self._predownscale_override)

        # Порядок шагов: ручной (drag) > выбранный авто/LLM > дефолт (ключ опущен).
        if self._user_order:
            config["order"] = list(self._user_order)
        elif self._order_override:
            config["order"] = list(self._order_override)

        return config

    def _get_selected_scale(self) -> int:
        btn = self.scale_group.checkedButton()
        return btn.property("scale") if btn else 4

    def _get_plugin_params(self, plugin_name: str, schema: dict) -> dict:
        params = {}
        for pname, pschema in schema.items():
            key = f"{plugin_name}.{pname}"
            if key in self._combos:
                params[pname] = self._combos[key].currentData()
            elif key in self._sliders:
                raw = self._sliders[key].value()
                if pschema.get("type") == "number":
                    params[pname] = raw / 100.0
                else:
                    params[pname] = raw
            else:
                params[pname] = pschema.get("default")
        return params

    def _apply_param_to_slider(self, plugin_name: str, pname: str, val):
        """Установить слайдер по схеме плагина (number -> x100, integer -> как есть)."""
        if not isinstance(val, (int, float)):
            return
        key = f"{plugin_name}.{pname}"
        slider = self._sliders.get(key)
        if slider is None:
            return
        plugin_cls = self._registry.get(plugin_name)
        schema = (getattr(plugin_cls, "params_schema", {}) or {}).get(pname, {})
        if schema.get("type") == "integer":
            slider.setValue(int(round(val)))
        else:
            slider.setValue(int(round(val * 100)))

    # --- Порядок шагов -----------------------------------------------------

    def _populate_order_list(self, order: list):
        self.order_list.clear()
        for token in order:
            item = QListWidgetItem(step_label(token))
            item.setData(Qt.ItemDataRole.UserRole, token)
            self.order_list.addItem(item)

    def get_order(self) -> list:
        """Текущая последовательность токенов в списке «Порядок обработки»."""
        return [self.order_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.order_list.count())]

    def _on_order_moved(self, *_):
        # Любое перетаскивание делает порядок ручным.
        self._user_order = self.get_order()

    def _move_order_item(self, delta: int):
        row = self.order_list.currentRow()
        new_row = row + delta
        if row < 0 or not (0 <= new_row < self.order_list.count()):
            return
        item = self.order_list.takeItem(row)
        self.order_list.insertItem(new_row, item)
        self.order_list.setCurrentRow(new_row)
        self._user_order = self.get_order()

    def _reset_order(self):
        """Сброс ручного порядка: показать LLM-порядок, если он был, иначе дефолт."""
        self._user_order = []
        self._populate_order_list(
            resolve_step_order(self._order_override or None))

    def get_max_refine_iterations(self) -> int:
        return self.max_iter_spin.value()

    def set_max_refine_iterations(self, value: int) -> None:
        self.max_iter_spin.setValue(max(1, min(10, int(value))))

    def is_ai_assistant_enabled(self) -> bool:
        """Whether the vision-model assistant is active for the auto buttons.

        When off, auto operations use only the algorithmic AutoConfigurator and
        skip the iterative re-evaluation loop.
        """
        return bool(self.ai_assistant_cb.isChecked())

    def set_ai_assistant_enabled(self, enabled: bool) -> None:
        self.ai_assistant_cb.setChecked(bool(enabled))

    def is_variants_enabled(self) -> bool:
        """«Сделай красиво» генерирует 4 стилевых варианта с выбором на
        каждой итерации (см. engine/four_variants.py)."""
        return bool(self.variants_cb.isChecked())

    def set_variants_enabled(self, enabled: bool) -> None:
        self.variants_cb.setChecked(bool(enabled))

    def is_predownscale_enabled(self) -> bool:
        """Разрешено ли авто-предуменьшение мыльных изображений."""
        return bool(self.predownscale_cb.isChecked())

    def set_predownscale_enabled(self, enabled: bool) -> None:
        self.predownscale_cb.setChecked(bool(enabled))

    def is_deblur_enabled(self) -> bool:
        """Whether the SmartDeblur checkbox in the single tab is ticked.

        Automatic operations consult this to decide if SmartDeblur may be used
        at all (the LLM/vision advisor is told not to emit deblur params when
        it is off).
        """
        cb = self._checkboxes.get("SmartDeblur")
        return bool(cb and cb.isChecked())

    def set_deblur_config(self, cfg: dict):
        """Store fine-tuning overrides from DeblurPanel and reflect the checkbox."""
        self._deblur_override = dict(cfg) if cfg else {}
        cb = self._checkboxes.get("SmartDeblur")
        if cb is not None:
            cb.setChecked(bool(cfg))

    def is_icedit_enabled(self) -> bool:
        """Whether the ICEdit checkbox is ticked.

        The vision/LLM advisor may propose an instruction edit only when this is
        on; otherwise automatic operations never consider ICEdit.
        """
        cb = self._checkboxes.get("ICEdit")
        return bool(cb and cb.isChecked())

    def set_icedit_config(self, cfg: dict):
        """Store the manual/auto ICEdit config and reflect the checkbox."""
        self._icedit_override = dict(cfg) if cfg else {}
        cb = self._checkboxes.get("ICEdit")
        if cb is not None:
            has = bool(cfg and (cfg.get("instruction") or "").strip())
            cb.setChecked(has)

    def is_face_enabled(self) -> bool:
        """Whether the CodeFormer face-restoration checkbox is ticked."""
        cb = self._checkboxes.get("CodeFormer")
        return bool(cb and cb.isChecked())

    def is_blend_enabled(self) -> bool:
        """Включено ли авто-смешивание вариантов."""
        return bool(self.blend_cb.isChecked())

    def set_blend_enabled(self, enabled: bool) -> None:
        self.blend_cb.setChecked(bool(enabled))

    def set_face_config(self, cfg: dict):
        """Store fine-tuning overrides from FacePanel and reflect the checkbox."""
        new = dict(cfg) if cfg else {}
        # Авто/LLM-конфиги не знают о ручных зонах: без ключа "regions"
        # сохраняем уже разметённые зоны, чтобы видимая разметка применялась.
        if "regions" not in new and self._face_override.get("regions"):
            new["regions"] = [list(z) for z in self._face_override["regions"]]
        self._face_override = new
        cb = self._checkboxes.get("CodeFormer")
        if cb is not None:
            cb.setChecked(bool(cfg.get("enabled", True)) if cfg else False)

    def apply_auto_config(self, config: dict):
        """Apply auto-configurator results to UI controls."""
        for cb in self._checkboxes.values():
            cb.setChecked(False)

        scale = config.get("scale", 4)
        for btn in self.scale_group.buttons():
            if btn.property("scale") == scale:
                btn.setChecked(True)
        self.enhance_only_cb.setChecked(config.get("enhance_only", False))

        up = config.get("upscale", {})
        idx = self.upscaler_combo.findData(up.get("plugin"))
        self.upscaler_combo.setCurrentIndex(idx if idx >= 0 else 0)

        for name, params in config.get("denoise", {}).items():
            if name in self._checkboxes:
                self._checkboxes[name].setChecked(True)
                if isinstance(params, dict):
                    for pname, val in params.items():
                        self._apply_param_to_slider(name, pname, val)

        for name, params in config.get("adjust", {}).items():
            if name in self._checkboxes:
                self._checkboxes[name].setChecked(True)
                if isinstance(params, dict):
                    for pname, val in params.items():
                        self._apply_param_to_slider(name, pname, val)

        # Reflect auto/LLM-chosen deblur on the SmartDeblur checkbox + override
        self.set_deblur_config(config.get("deblur") or {})
        self.set_icedit_config(config.get("icedit") or {})
        self.set_face_config(config.get("face") or {})
        self._predownscale_override = dict(config.get("predownscale") or {})

        # Remember the chosen step order so a later manual "Обработать" reuses it.
        order = config.get("order")
        self._order_override = list(order) if isinstance(order, list) else []
        self._user_order = []  # авто-порядок не помечается ручным
        self._populate_order_list(resolve_step_order(order))

        self.preset_combo.setCurrentIndex(0)  # «(Вручную)» — всегда индекс 0

    def _on_preset_changed(self, name: str):
        # Индекс 0 — сентинел «(Вручную)»; независим от языка/текста.
        if self.preset_combo.currentIndex() == 0:
            return
        preset = self._preset_loader.load(name)
        if not preset:
            return
        self._apply_preset(preset)

    def _apply_preset(self, preset: dict):
        for cb in self._checkboxes.values():
            cb.setChecked(False)
        self._order_override = []  # presets use the default processing order
        self._user_order = []
        self._predownscale_override = {}
        self._populate_order_list(list(PipelineExecutor.DEFAULT_ORDER))

        scale = preset.get("scale", 4)
        for btn in self.scale_group.buttons():
            if btn.property("scale") == scale:
                btn.setChecked(True)

        self.enhance_only_cb.setChecked(preset.get("enhance_only", False))

        pipeline = preset.get("pipeline", {})
        scale_cfg = pipeline.get("scale", {})
        plugin_name = scale_cfg.get("plugin")
        idx = self.upscaler_combo.findData(plugin_name)
        self.upscaler_combo.setCurrentIndex(idx if idx >= 0 else 0)

        for name in pipeline.get("denoise", {}):
            if name in self._checkboxes:
                self._checkboxes[name].setChecked(True)

        for name in pipeline.get("adjust", {}):
            if name in self._checkboxes:
                self._checkboxes[name].setChecked(True)
