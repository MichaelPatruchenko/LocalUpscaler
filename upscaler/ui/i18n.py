"""Лёгкий слой интернационализации (RU/EN) с ретрансляцией на лету.

Строки хранятся в плоских таблицах _STRINGS[lang][key]. tr() никогда не
падает: нет ключа в текущем языке -> берём из другого -> возвращаем сам key.
Персистентность языка — через config.load_settings/save_settings.
"""
from PySide6.QtCore import QObject, Signal

_LANGUAGES = ("ru", "en")

# Плоские таблицы переводов. Ключи — стабильные слаги "domain.name".
# Пополняются задачами этапа D; паритет ключей проверяется тестом.
_STRINGS = {
    "ru": {
        # --- Кнопки действий ---
        "btn.process": "Обработать",
        "btn.auto": "Авто (умные настройки)",
        "btn.make_beautiful": "Сделай красиво",
        "btn.upload_image": "Загрузить изображение",
        "btn.upload_dir": "Загрузить папку",
        # --- Общие статусы ---
        "status.zone_count": "Зон: {n}",
        "status.analyzing": "Анализ изображения...",
        # --- Язык ---
        "lang.toggle_tooltip": "Переключить язык интерфейса (RU/EN)",
        # --- Заголовок окна ---
        "win.title": "Upscaler — Улучшение изображений",
        # --- Режимы (радио-кнопки левой панели) ---
        "mode.single": "Одиночный",
        "mode.batch": "Пакетный",
        "mode.colorize": "Раскрашивание",
        "mode.deblur": "SmartDeblur",
        "mode.icedit": "ICEdit",
        "mode.face": "Лица",
        "mode.blend": "Смешивание",
        # --- Заголовки секций ControlPanel ---
        "section.preset_scale": "Пресет и масштаб",
        "section.upscaler": "Апскейлер",
        "section.denoise": "Шумоподавление",
        "section.adjust": "Коррекция",
        "section.adjust_tone": "Тон",
        "section.adjust_color": "Цвет",
        "section.adjust_detail": "Детали",
        "section.restore": "Восстановление",
        "section.edit": "Редактирование",
        "section.blend": "Смешивание",
        "section.ai": "ИИ-ассистент и доработка",
        "section.order": "Порядок обработки",
        # --- Литералы панели ControlPanel ---
        "panel.preset_label": "Пресет:",
        "panel.preset_manual": "(Вручную)",
        "panel.scale_group_title": "Масштаб",
        "panel.enhance_only": "Только улучшение (без масштабирования)",
        "panel.predownscale": "Авто-уменьшение мыльных изображений",
        "panel.predownscale_tooltip": (
            "Если изображение большое, но реальной детализации в нём меньше "
            "номинала (апсемпл, «мыло»), авто-режим сначала уменьшит его до "
            "эффективного разрешения и лишь потом обработает."),
        "panel.upscaler_none": "(Нет — без масштабирования)",
        "panel.face_restore": "Восстановление лиц (CodeFormer)",
        "panel.restore_hint": ("Параметры рассчитываются автоматически.\n"
                               "Тонкая настройка — на вкладке «SmartDeblur»."),
        "panel.edit_hint": ("Инструкцию подбирает vision-модель.\n"
                            "Ручной ввод — на вкладке «ICEdit»."),
        "panel.blend_checkbox": "Смешивание вариантов",
        "panel.blend_hint": (
            "Автоматически накладывает промежуточные варианты обработки "
            "друг на друга (режимы Photoshop), если это улучшает итоговую "
            "метрику качества. Ручной режим — на вкладке «Смешивание»."),
        "panel.ai_assistant_checkbox": "ИИ-ассистент (vision-модель)",
        "panel.ai_hint": (
            "Снимите, чтобы автоматические кнопки использовали только "
            "алгоритмический подбор без vision-модели и без переоценки."),
        "panel.max_iter_label": "Макс. повторных обработок:",
        "panel.max_iter_tooltip": (
            "Сколько конвейеров максимум формирует vision-модель: после "
            "обработки результат оценивается и при необходимости дорабатывается "
            "(1 = один проход без оценки; N = до N−1 повторных обработок)."),
        "panel.order_up": "Вверх",
        "panel.order_down": "Вниз",
        "panel.order_reset": "Сброс",
        "panel.order_hint": (
            "Перетащите шаги (или кнопками), чтобы изменить порядок применения. "
            "Выполняются только включённые шаги. «Сброс» — вернуть авто-порядок."),
        # --- Подписи шагов конвейера (engine/pipeline.py STEP_LABELS) ---
        "step.predownscale": "Предуменьшение",
        "step.denoise": "Шумоподавление",
        "step.deblur": "Деблюр (SmartDeblur)",
        "step.icedit": "Редактирование (ICEdit)",
        "step.optics": "Оптика (виньетка/ХА)",
        "step.white_balance": "Баланс белого",
        "step.dehaze": "Удаление дымки",
        "step.auto_color": "Авто-цвет",
        "step.auto_levels": "Авто-уровни",
        "step.auto_tone": "Авто-тон",
        "step.auto_contrast": "Авто-контраст",
        "step.brightness": "Яркость",
        "step.contrast": "Контраст",
        "step.shadows_highlights": "Тени/Света",
        "step.dodge_burn": "Осветление/Затемнение",
        "step.saturation": "Насыщенность",
        "step.vibrance": "Сочность",
        "step.split_tone": "Раздельное тонирование",
        "step.clarity": "Локальный контраст",
        "step.sharpness": "Резкость",
        "step.adjust": "Прочие коррекции",
        "step.upscale": "Масштабирование",
        "step.face_restore": "Лица (CodeFormer)",
        "step.skin_smooth": "Ретушь кожи",
        "step.refocus": "Рефокус",
        "step.sharpen": "Резкость (пост)",
        "step.colorize": "Раскрашивание",
        "step.blend": "Смешивание вариантов",
        # --- Подписи режимов наложения (engine/blend.py BLEND_MODE_LABELS) ---
        "blend.normal": "Обычный",
        "blend.dissolve": "Растворение",
        "blend.darken": "Затемнение",
        "blend.multiply": "Умножение",
        "blend.color_burn": "Затемнение основы",
        "blend.linear_burn": "Линейный затемнитель",
        "blend.lighten": "Замена светлым",
        "blend.screen": "Экран",
        "blend.color_dodge": "Осветление основы",
        "blend.linear_dodge": "Линейный осветлитель",
        "blend.overlay": "Перекрытие",
        "blend.soft_light": "Мягкий свет",
        "blend.hard_light": "Жёсткий свет",
        "blend.vivid_light": "Яркий свет",
        "blend.linear_light": "Линейный свет",
        "blend.pin_light": "Точечный свет",
        "blend.hard_mix": "Жёсткое смешение",
        "blend.difference": "Разница",
        "blend.exclusion": "Исключение",
        "blend.subtract": "Вычитание",
        "blend.divide": "Разделение",
        "blend.hue": "Цветовой тон",
        "blend.saturation": "Насыщенность",
        "blend.color": "Цветность",
        "blend.luminosity": "Яркость",
        # --- Фрагменты AutoConfigurator.describe() ---
        "desc.predownscale": "Уменьшение: до {pct}%",
        "desc.denoise": "Шумоподавление: {names}",
        "desc.correction": "Коррекция: {names}",
        "desc.scale": "Масштаб: {plugin} ({mode})",
        "desc.scale_mode_enhance": "улучшение",
        "desc.sharpness": "Резкость: {parts}",
        "desc.deblur": "Деблюр: SmartDeblur (авто)",
        "desc.face": "Лица: CodeFormer (w={fidelity})",
        "desc.order": "Порядок: {labels}",
        "desc.none": "Без обработки",
        # --- BatchPanel ---
        "batch.add_files": "Добавить файлы",
        "batch.add_dir": "Добавить папку",
        "batch.clear": "Очистить",
        "batch.file_count": "{n} файлов",
        "batch.output_caption": "Выход:",
        "batch.output_default": "(как у исходных)",
        "batch.browse": "Обзор",
        "batch.process_btn": "Обработать (настройки)",
        "batch.process_tooltip": "Применить текущие настройки из панели к каждому изображению",
        "batch.beautiful_btn": "Сделай красиво",
        "batch.beautiful_tooltip": "Авто-анализ и оптимальная обработка для каждого изображения",
        "batch.progress": "Обработка {current} из {total}: {filename}",
        "batch.select_images_title": "Выбрать изображения",
        "batch.images_filter": "Изображения (*.png *.jpg *.jpeg *.tiff *.bmp *.webp)",
        "batch.select_dir_title": "Выбрать папку",
        "batch.output_dir_title": "Выходная папка",
        # --- ColorizePanel ---
        "colorize.title": "Раскрашивание",
        "colorize.model_group": "Модель",
        "colorize.model_ddcolor": "DDColor (только фото)",
        "colorize.model_deoldify": "DeOldify (фото и видео)",
        "colorize.model_colormnet": "ColorMNet (фото и видео)",
        "colorize.variant_label": "Вариант:",
        "colorize.ddcolor_artistic": "Artistic (рекомендуется)",
        "colorize.ddcolor_modelscope": "ModelScope",
        "colorize.deoldify_stable": "Stable (стабильная)",
        "colorize.deoldify_artistic": "Artistic (художественная)",
        "colorize.deoldify_video": "Video (для видео)",
        "colorize.params_group": "Параметры",
        "colorize.strength_label": "Сила:",
        "colorize.photo_group": "Фото",
        "colorize.photo_btn": "Раскрасить текущее фото",
        "colorize.video_group": "Видео",
        "colorize.no_video": "Видео не выбрано",
        "colorize.select_video": "Выбрать видео",
        "colorize.video_btn": "Раскрасить видео",
        "colorize.frame_progress": "Кадр {current} из {total}",
        "colorize.video_filter": "Видео (*.mp4 *.avi *.mkv *.mov *.webm);;Все файлы (*)",
        "colorize.save_video_title": "Сохранить видео",
        "colorize.video_save_filter": "MP4 (*.mp4);;AVI (*.avi);;Все файлы (*)",
        "colorize.info_ddcolor": (
            "Высокое качество раскрашивания фотографий с использованием двойных "
            "декодеров. Два варианта: Artistic (более яркие цвета) и ModelScope "
            "(более реалистичные). Только для фото. ~900 МБ VRAM."),
        "colorize.info_deoldify": (
            "Классическая модель для раскрашивания фото и видео. Три варианта: "
            "Stable (стабильная), Artistic (художественная), Video (для видео). "
            "~600 МБ VRAM."),
        "colorize.info_colormnet": (
            "Раскрашивание фото и видео с временной согласованностью кадров. "
            "Лучший выбор для видео. ~700 МБ VRAM."),
        # --- DeblurPanel ---
        "deblur.title": "SmartDeblur — восстановление резкости",
        "deblur.mode_group": "Режим",
        "deblur.mode_auto": "Авто",
        "deblur.mode_manual": "Ручной",
        "deblur.params_group": "Параметры",
        "deblur.blur_type_label": "Тип размытия:",
        "deblur.blur_focus": "Расфокус",
        "deblur.blur_motion": "Смаз (движение)",
        "deblur.blur_gaussian": "Гаусс",
        "deblur.radius": "Радиус",
        "deblur.angle": "Угол",
        "deblur.smooth": "Сглаживание",
        "deblur.feather": "Размытие края",
        "deblur.correction": "Краевая коррекция",
        "deblur.method_label": "Метод:",
        "deblur.method_wiener": "Wiener (быстро)",
        "deblur.method_tikhonov": "Tikhonov",
        "deblur.method_tv": "Total Variation (качество)",
        "deblur.method_rl": "Richardson-Lucy",
        "deblur.iterations": "Итерации",
        "deblur.edge_taper": "Подавление звона (edge taper)",
        "deblur.preview_btn": "Предпросмотр",
        # --- ICEditPanel ---
        "icedit.title": "ICEdit — редактирование по инструкции",
        "icedit.instruction_label": "Инструкция (на английском):",
        "icedit.instruction_placeholder": (
            "например: make the hair green / remove the watermark"),
        "icedit.params_group": "Параметры",
        "icedit.lora_label": "LoRA:",
        "icedit.lora_normal": "Normal LoRA (рекомендуется)",
        "icedit.lora_moe": "MoE-LoRA (экспериментально)",
        "icedit.steps_label": "Шаги",
        "icedit.guidance_label": "Guidance",
        "icedit.seed_label": "Seed (-1 = случайно):",
        "icedit.quant_label": "Квантование:",
        "icedit.quant_q4": "Q4 (меньше VRAM)",
        "icedit.quant_q5": "Q5 (качество)",
        "icedit.vram_label": "VRAM (offload):",
        "icedit.offload_model": "Model offload (по умолчанию)",
        "icedit.offload_none": "Без offload (быстро, много VRAM)",
        "icedit.offload_sequential": "Sequential (минимум VRAM)",
        "icedit.preview_btn": "Предпросмотр",
        # --- FacePanel ---
        "face.title": "Восстановление лиц — CodeFormer",
        "face.params_group": "Параметры",
        "face.fidelity_label": "Верность (fidelity):",
        "face.fidelity_hint": "Выше — ближе к оригиналу; ниже — сильнее реставрация.",
        "face.minpx_label": "Мин. размер лица (px):",
        "face.bg_checkbox": "Улучшать фон",
        "face.zones_group": "Зоны лиц (ручная разметка)",
        "face.zone_edit_btn": "Режим разметки",
        "face.delete_zone": "Удалить",
        "face.clear_zones": "Очистить",
        "face.zones_hint": (
            "Включите режим разметки и нарисуйте рамки на холсте (ЛКМ). "
            "Размеченные зоны заменяют автопоиск лиц; без разметки лица "
            "ищутся автоматически. Обработка выполняется только при "
            "включённом «Восстановлении лиц» на его шаге конвейера."),
        "face.preview_btn": "Предпросмотр",
        "face.zone_item": "Зона {i}: {w}%x{h}% @ ({x}%, {y}%)",
        # --- BlendPanel (собственные литералы; blend.normal..luminosity — режимы наложения выше) ---
        "blend.panel_title": "Смешивание вариантов — ручной режим",
        "blend.variants": "Варианты",
        "blend.mode_caption": "Режим:",
        "blend.opacity_caption": "Прозрачность:",
        "blend.blend_selected": "Смешать выбранные",
        "blend.auto_btn": "Авто-подбор",
        "blend.preview_btn": "Предпросмотр",
        "blend.apply_btn": "Применить (в историю)",
        "blend.result_label": "Результат смешивания",
        "blend.auto_result_label": "Авто-подбор: результат",
        "blend.hint": (
            "Выберите два варианта в списке (ЛКМ — первый, ПКМ — второй), "
            "режим наложения и прозрачность. «Смешать выбранные» добавляет "
            "результат новым вариантом для дальнейшего комбинирования; "
            "«Применить» сохраняет его в историю. «Авто-подбор» ищет лучшую "
            "комбинацию по метрике качества среди всех вариантов и сразу "
            "показывает результат."),
        # --- HistoryPanel ---
        "history.title": "История",
        "history.revert": "Откат",
        "history.compare": "Сравнить",
        "history.delete": "Удалить",
        "history.info_group": "Информация",
        "history.info_none": "Ничего не выбрано",
        # --- SettingsDialog ---
        "settings.window_title": "Настройки",
        "settings.gpu_group": "GPU / Производительность",
        "settings.cuda_available": "Доступен",
        "settings.cuda_unavailable": "Недоступен",
        "settings.cuda_na": "Н/Д",
        "settings.info_text": "PyTorch {ver}  |  CUDA: {status} (драйвер: {cudaver})",
        "settings.cuda_warning": (
            "CUDA недоступен. Установите PyTorch с CUDA:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"),
        "settings.device_label": "Устройство вычислений:",
        "settings.gpu_option": "GPU {i}: {name} ({mem} ГБ)",
        "settings.tile_size_label": "Размер тайла:",
        "settings.tile_size_tooltip": "Большие тайлы используют больше VRAM, но обрабатываются быстрее",
        "settings.tile_overlap_label": "Перекрытие тайлов:",
        "settings.tile_overlap_tooltip": "Перекрытие между тайлами для уменьшения артефактов на швах",
        "settings.gpu_denoise_checkbox": (
            "Заменять BM3D на GPU-шумодав (SCUNet) при наличии видеокарты"),
        "settings.gpu_denoise_tooltip": (
            "BM3D работает только на CPU и медленно. При наличии CUDA выбранный "
            "BM3D будет заменён на быстрый GPU-шумодав SCUNet. Снимите галочку, "
            "чтобы всегда использовать именно BM3D."),
        "settings.output_group": "Вывод",
        "settings.format_label": "Формат по умолчанию:",
        "settings.output_dir_placeholder": "Та же, что и у исходного",
        "settings.browse": "Обзор...",
        "settings.output_dir_label": "Папка вывода:",
        "settings.history_group": "История",
        "settings.max_entries_label": "Макс. записей:",
        "settings.retention_label": "Хранение:",
        "settings.retention_suffix": " дней",
        "settings.storage_group": "Хранилище",
        "settings.cache_dir_placeholder": "~/.upscaler/models",
        "settings.cache_dir_label": "Кэш моделей:",
        "settings.appearance_group": "Внешний вид",
        "settings.theme_label": "Тема:",
        "settings.icedit_group": "ICEdit",
        "settings.icedit_offload_label": "Offload памяти:",
        "settings.icedit_offload_model": "Model offload (по умолчанию)",
        "settings.icedit_offload_none": "Без offload (много VRAM)",
        "settings.icedit_offload_sequential": "Sequential (минимум VRAM)",
        "settings.icedit_quant_label": "Квантование:",
        "settings.icedit_quant_q4": "Q4 (меньше VRAM)",
        "settings.icedit_quant_q5": "Q5 (качество)",
        "settings.output_dir_dialog_title": "Выбор папки вывода",
        "settings.cache_dir_dialog_title": "Выбор папки кэша моделей",
        # --- DownloadDialog ---
        "download.title": "Загрузка {model}",
        "download.progress_label": "Загрузка {model}...",
        "download.cancel": "Отмена",
        # --- CompareDialog ---
        "compare.window_title": "Сравнение — Оригинал vs Результат",
        "compare.original_label": "Оригинал",
        "compare.result_label": "Результат",
        "compare.close_btn": "Закрыть",
        # --- Меню (MainWindow) ---
        "menu.file": "&Файл",
        "menu.open": "&Открыть изображение...",
        "menu.save": "&Сохранить результат...",
        "menu.exit": "В&ыход",
        "menu.edit": "&Редактирование",
        "menu.settings": "&Настройки...",
        "menu.clear_history": "Очистить &историю",
        "menu.view": "&Вид",
        "menu.fit_view": "&По размеру окна",
        "menu.processing": "&Обработка",
        "menu.process_action": "&Обработать",
        "menu.cancel_action": "&Отмена",
        "menu.help": "&Справка",
        "menu.about": "&О программе",
        # --- Сообщения (QMessageBox) ---
        "msg.about_title": "О программе Upscaler",
        "msg.about_body": (
            "Upscaler — Профессиональное улучшение изображений\n\n"
            "ИИ-модели: Real-ESRGAN, HAT-S, SwinIR, OmniSR, DAT\n"
            "Традиционные: Lanczos, Bicubic, Sinc, NEDI, DCCI, EGGI/SIRE\n"
            "Шумоподавление: BM3D, SCUNet, NAFNet, Wavelet, NL-Means, Bilateral\n"
            "Раскрашивание: DDColor, DeOldify, ColorMNet"),
        "msg.resume_session_title": "Возобновить сессию",
        "msg.resume_session_body": "Найдено сессий: {n}. Возобновить последнюю?",
        "msg.error_title": "Ошибка",
        "msg.load_error_body": "Не удалось загрузить изображение:\n{error}",
        "msg.no_image_title": "Нет изображения",
        "msg.no_image_body": "Сначала загрузите изображение.",
        "msg.blend_title": "Смешивание",
        "msg.blend_sources_unavailable": "Не все источники рецепта доступны.",
        "msg.blend_need_two_variants": (
            "Нужно минимум два варианта (обработайте изображение, "
            "чтобы появились варианты шагов конвейера)."),
        "msg.no_instruction_title": "Нет инструкции",
        "msg.no_instruction_body": "Введите инструкцию для ICEdit.",
        "msg.result_load_error_body": "Не удалось загрузить результат:\n{error}",
        "msg.processing_error_title": "Ошибка обработки",
        "msg.processing_error_body": "Этап: {stage}\n\n{error}",
        "msg.no_files_title": "Нет файлов",
        "msg.no_files_body": "Добавьте файлы для пакетной обработки.",
        "msg.clear_history_title": "Очистить историю",
        "msg.clear_history_body": "Удалить всю историю? Это действие необратимо.",
        # --- Диалоги файлов (MainWindow) ---
        "dialog.open_image_title": "Открыть изображение",
        "dialog.images_filter": "Изображения ({exts});;Все файлы (*)",
        "dialog.open_dir_title": "Выбрать папку с изображениями",
        "dialog.save_result_title": "Сохранить результат",
        "dialog.save_filter": "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff);;Все файлы (*)",
        # --- Подсказки (MainWindow) ---
        "tooltip.auto_btn": "Анализировать изображение и автоматически подобрать настройки",
        "tooltip.make_beautiful_btn": "Автоматический анализ + подбор параметров + обработка",
        # --- Тулбар зон лиц (MainWindow) ---
        "zones.toolbar_label": "Зоны лиц",
        "zones.toolbar_tooltip": (
            "Режим разметки лиц: рисуйте рамки на холсте (ЛКМ). "
            "Зоны заменяют автопоиск лиц для CodeFormer."),
        "zones.clear_btn": "Очистить",
        # --- Статус-бар и прогресс (MainWindow) ---
        "status.ready": "Готов",
        "status.device_cpu_no_torch": "Устройство: CPU (torch не установлен)",
        "status.device_cpu": "Устройство: CPU",
        "status.device_gpu": "Устройство: {name}",
        "status.device_cpu_no_cuda": "Устройство: CPU (CUDA недоступен)",
        "status.disk_gb_warning": "История: {gb:.1f} ГБ (ВНИМАНИЕ)",
        "status.disk_gb": "История: {gb:.1f} ГБ",
        "status.disk_mb": "История: {mb:.0f} МБ",
        "status.loaded": "Загружено: {name} ({w}x{h})",
        "status.llm_refining": "LLM уточняет параметры обработки...",
        "status.auto_applied": "Авто: {desc}",
        "status.blend_preview_ready": "Смешивание: предпросмотр готов",
        "status.blend_saved": "Смешивание: результат сохранён как версия {version}",
        "status.blend_variant_added": "Смешивание: результат добавлен как вариант {id}",
        "status.blend_auto_result": "Авто-подбор: база {base}, слоёв: {n} (score {score:.3f})",
        "status.auto_processing": "Авто: {desc} — обработка...",
        "status.progress": "[{stage}] {message}",
        "status.complete_metrics": "Готово | BRISQUE: {brisque:.1f} | NIQE: {niqe:.1f}",
        "status.done": "Готово",
        "status.evaluating": "Оценка результата (итерация {iter}/{max})...",
        "status.refining": "Доработка (итерация {iter}/{max})...",
        "status.error_stage": "Ошибка: {stage}",
        "status.cancelled": "Отменено",
        "status.reverted": "Откат к оригиналу",
        "status.batch_done": "Завершено: {n} файлов",
        "status.batch_complete": "Пакетная обработка завершена",
        "status.colorizing_photo": "Раскрашивание с {model}...",
        "status.colorizing_video": "Раскрашивание видео с {model}...",
        "status.video_done": "Готово: {path}",
        "status.video_error": "Ошибка: {error}",
        "status.saved": "Сохранено: {path}",
    },
    "en": {
        "btn.process": "Process",
        "btn.auto": "Auto (smart settings)",
        "btn.make_beautiful": "Make it beautiful",
        "btn.upload_image": "Load image",
        "btn.upload_dir": "Load folder",
        "status.zone_count": "Zones: {n}",
        "status.analyzing": "Analyzing image...",
        "lang.toggle_tooltip": "Switch interface language (RU/EN)",
        "win.title": "Upscaler — Image Enhancement",
        "mode.single": "Single",
        "mode.batch": "Batch",
        "mode.colorize": "Colorize",
        "mode.deblur": "SmartDeblur",
        "mode.icedit": "ICEdit",
        "mode.face": "Faces",
        "mode.blend": "Blend",
        "section.preset_scale": "Preset & Scale",
        "section.upscaler": "Upscaler",
        "section.denoise": "Denoising",
        "section.adjust": "Correction",
        "section.adjust_tone": "Tone",
        "section.adjust_color": "Color",
        "section.adjust_detail": "Details",
        "section.restore": "Restoration",
        "section.edit": "Editing",
        "section.blend": "Blending",
        "section.ai": "AI assistant & refinement",
        "section.order": "Processing order",
        "panel.preset_label": "Preset:",
        "panel.preset_manual": "(Manual)",
        "panel.scale_group_title": "Scale",
        "panel.enhance_only": "Enhance only (no upscaling)",
        "panel.predownscale": "Auto-downscale soft images",
        "panel.predownscale_tooltip": (
            "If the image is large but its real detail is below nominal "
            "(upsampled, \"soft\"), auto mode first downscales it to the "
            "effective resolution and only then processes it."),
        "panel.upscaler_none": "(None — no upscaling)",
        "panel.face_restore": "Face restoration (CodeFormer)",
        "panel.restore_hint": ("Parameters are computed automatically.\n"
                               "Fine-tuning — on the \"SmartDeblur\" tab."),
        "panel.edit_hint": ("The instruction is chosen by the vision model.\n"
                            "Manual input — on the \"ICEdit\" tab."),
        "panel.blend_checkbox": "Variant blending",
        "panel.blend_hint": (
            "Automatically layers intermediate processing variants on top of "
            "each other (Photoshop blend modes) if it improves the final "
            "quality metric. Manual mode — on the \"Blend\" tab."),
        "panel.ai_assistant_checkbox": "AI assistant (vision model)",
        "panel.ai_hint": (
            "Uncheck so automatic buttons use only algorithmic selection, "
            "without the vision model and without re-evaluation."),
        "panel.max_iter_label": "Max refinement passes:",
        "panel.max_iter_tooltip": (
            "How many pipelines the vision model builds at most: after "
            "processing, the result is evaluated and refined if needed "
            "(1 = single pass without evaluation; N = up to N-1 refinement "
            "passes)."),
        "panel.order_up": "Up",
        "panel.order_down": "Down",
        "panel.order_reset": "Reset",
        "panel.order_hint": (
            "Drag steps (or use the buttons) to change the application "
            "order. Only enabled steps run. \"Reset\" restores the automatic "
            "order."),
        "step.predownscale": "Pre-downscale",
        "step.denoise": "Denoise",
        "step.deblur": "Deblur (SmartDeblur)",
        "step.icedit": "Edit (ICEdit)",
        "step.optics": "Optics (vignette/CA)",
        "step.white_balance": "White Balance",
        "step.dehaze": "Dehaze",
        "step.auto_color": "Auto Color",
        "step.auto_levels": "Auto Levels",
        "step.auto_tone": "Auto Tone",
        "step.auto_contrast": "Auto Contrast",
        "step.brightness": "Brightness",
        "step.contrast": "Contrast",
        "step.shadows_highlights": "Shadows/Highlights",
        "step.dodge_burn": "Dodge & Burn",
        "step.saturation": "Saturation",
        "step.vibrance": "Vibrance",
        "step.split_tone": "Split Toning",
        "step.clarity": "Clarity",
        "step.sharpness": "Sharpness",
        "step.adjust": "Other adjustments",
        "step.upscale": "Upscale",
        "step.face_restore": "Faces (CodeFormer)",
        "step.skin_smooth": "Skin smoothing",
        "step.refocus": "Refocus",
        "step.sharpen": "Sharpen (post)",
        "step.colorize": "Colorize",
        "step.blend": "Variant blending",
        "blend.normal": "Normal",
        "blend.dissolve": "Dissolve",
        "blend.darken": "Darken",
        "blend.multiply": "Multiply",
        "blend.color_burn": "Color Burn",
        "blend.linear_burn": "Linear Burn",
        "blend.lighten": "Lighten",
        "blend.screen": "Screen",
        "blend.color_dodge": "Color Dodge",
        "blend.linear_dodge": "Linear Dodge",
        "blend.overlay": "Overlay",
        "blend.soft_light": "Soft Light",
        "blend.hard_light": "Hard Light",
        "blend.vivid_light": "Vivid Light",
        "blend.linear_light": "Linear Light",
        "blend.pin_light": "Pin Light",
        "blend.hard_mix": "Hard Mix",
        "blend.difference": "Difference",
        "blend.exclusion": "Exclusion",
        "blend.subtract": "Subtract",
        "blend.divide": "Divide",
        "blend.hue": "Hue",
        "blend.saturation": "Saturation",
        "blend.color": "Color",
        "blend.luminosity": "Luminosity",
        "desc.predownscale": "Downscale: to {pct}%",
        "desc.denoise": "Denoising: {names}",
        "desc.correction": "Correction: {names}",
        "desc.scale": "Scale: {plugin} ({mode})",
        "desc.scale_mode_enhance": "enhance",
        "desc.sharpness": "Sharpness: {parts}",
        "desc.deblur": "Deblur: SmartDeblur (auto)",
        "desc.face": "Faces: CodeFormer (w={fidelity})",
        "desc.order": "Order: {labels}",
        "desc.none": "No processing",
        # --- BatchPanel ---
        "batch.add_files": "Add files",
        "batch.add_dir": "Add folder",
        "batch.clear": "Clear",
        "batch.file_count": "{n} files",
        "batch.output_caption": "Output:",
        "batch.output_default": "(same as source)",
        "batch.browse": "Browse",
        "batch.process_btn": "Process (settings)",
        "batch.process_tooltip": "Apply the current panel settings to each image",
        "batch.beautiful_btn": "Make it beautiful",
        "batch.beautiful_tooltip": "Auto-analyze and optimally process each image",
        "batch.progress": "Processing {current} of {total}: {filename}",
        "batch.select_images_title": "Select images",
        "batch.images_filter": "Images (*.png *.jpg *.jpeg *.tiff *.bmp *.webp)",
        "batch.select_dir_title": "Select folder",
        "batch.output_dir_title": "Output folder",
        # --- ColorizePanel ---
        "colorize.title": "Colorize",
        "colorize.model_group": "Model",
        "colorize.model_ddcolor": "DDColor (photo only)",
        "colorize.model_deoldify": "DeOldify (photo & video)",
        "colorize.model_colormnet": "ColorMNet (photo & video)",
        "colorize.variant_label": "Variant:",
        "colorize.ddcolor_artistic": "Artistic (recommended)",
        "colorize.ddcolor_modelscope": "ModelScope",
        "colorize.deoldify_stable": "Stable",
        "colorize.deoldify_artistic": "Artistic",
        "colorize.deoldify_video": "Video (for video)",
        "colorize.params_group": "Parameters",
        "colorize.strength_label": "Strength:",
        "colorize.photo_group": "Photo",
        "colorize.photo_btn": "Colorize current photo",
        "colorize.video_group": "Video",
        "colorize.no_video": "No video selected",
        "colorize.select_video": "Select video",
        "colorize.video_btn": "Colorize video",
        "colorize.frame_progress": "Frame {current} of {total}",
        "colorize.video_filter": "Video (*.mp4 *.avi *.mkv *.mov *.webm);;All files (*)",
        "colorize.save_video_title": "Save video",
        "colorize.video_save_filter": "MP4 (*.mp4);;AVI (*.avi);;All files (*)",
        "colorize.info_ddcolor": (
            "High-quality photo colorization using dual decoders. Two "
            "variants: Artistic (more vivid colors) and ModelScope (more "
            "realistic). Photo only. ~900 MB VRAM."),
        "colorize.info_deoldify": (
            "Classic model for photo and video colorization. Three "
            "variants: Stable, Artistic, Video (for video). ~600 MB VRAM."),
        "colorize.info_colormnet": (
            "Photo and video colorization with temporal frame consistency. "
            "Best choice for video. ~700 MB VRAM."),
        # --- DeblurPanel ---
        "deblur.title": "SmartDeblur — sharpness restoration",
        "deblur.mode_group": "Mode",
        "deblur.mode_auto": "Auto",
        "deblur.mode_manual": "Manual",
        "deblur.params_group": "Parameters",
        "deblur.blur_type_label": "Blur type:",
        "deblur.blur_focus": "Defocus",
        "deblur.blur_motion": "Motion blur",
        "deblur.blur_gaussian": "Gaussian",
        "deblur.radius": "Radius",
        "deblur.angle": "Angle",
        "deblur.smooth": "Smoothing",
        "deblur.feather": "Edge feather",
        "deblur.correction": "Edge correction",
        "deblur.method_label": "Method:",
        "deblur.method_wiener": "Wiener (fast)",
        "deblur.method_tikhonov": "Tikhonov",
        "deblur.method_tv": "Total Variation (quality)",
        "deblur.method_rl": "Richardson-Lucy",
        "deblur.iterations": "Iterations",
        "deblur.edge_taper": "Ringing suppression (edge taper)",
        "deblur.preview_btn": "Preview",
        # --- ICEditPanel ---
        "icedit.title": "ICEdit — instruction-based editing",
        "icedit.instruction_label": "Instruction (in English):",
        "icedit.instruction_placeholder": (
            "e.g.: make the hair green / remove the watermark"),
        "icedit.params_group": "Parameters",
        "icedit.lora_label": "LoRA:",
        "icedit.lora_normal": "Normal LoRA (recommended)",
        "icedit.lora_moe": "MoE-LoRA (experimental)",
        "icedit.steps_label": "Steps",
        "icedit.guidance_label": "Guidance",
        "icedit.seed_label": "Seed (-1 = random):",
        "icedit.quant_label": "Quantization:",
        "icedit.quant_q4": "Q4 (less VRAM)",
        "icedit.quant_q5": "Q5 (quality)",
        "icedit.vram_label": "VRAM (offload):",
        "icedit.offload_model": "Model offload (default)",
        "icedit.offload_none": "No offload (fast, high VRAM)",
        "icedit.offload_sequential": "Sequential (minimum VRAM)",
        "icedit.preview_btn": "Preview",
        # --- FacePanel ---
        "face.title": "Face restoration — CodeFormer",
        "face.params_group": "Parameters",
        "face.fidelity_label": "Fidelity:",
        "face.fidelity_hint": "Higher — closer to the original; lower — stronger restoration.",
        "face.minpx_label": "Min face size (px):",
        "face.bg_checkbox": "Enhance background",
        "face.zones_group": "Face zones (manual)",
        "face.zone_edit_btn": "Marking mode",
        "face.delete_zone": "Delete",
        "face.clear_zones": "Clear",
        "face.zones_hint": (
            "Enable marking mode and draw boxes on the canvas (left click). "
            "Marked zones replace automatic face detection; without marking, "
            "faces are found automatically. Processing only runs when "
            "\"Face restoration\" is enabled on its pipeline step."),
        "face.preview_btn": "Preview",
        "face.zone_item": "Zone {i}: {w}%x{h}% @ ({x}%, {y}%)",
        # --- BlendPanel ---
        "blend.panel_title": "Variant blending — manual mode",
        "blend.variants": "Variants",
        "blend.mode_caption": "Mode:",
        "blend.opacity_caption": "Opacity:",
        "blend.blend_selected": "Blend selected",
        "blend.auto_btn": "Auto-select",
        "blend.preview_btn": "Preview",
        "blend.apply_btn": "Apply (to history)",
        "blend.result_label": "Blend result",
        "blend.auto_result_label": "Auto-select result",
        "blend.hint": (
            "Select two variants in the list (left click = first, right "
            "click = second), pick a blend mode and opacity. \"Blend "
            "selected\" adds the result as a new variant for further "
            "combining; \"Apply\" saves it to history. \"Auto-select\" "
            "searches for the best combination by quality metric across "
            "all variants and shows the result immediately."),
        # --- HistoryPanel ---
        "history.title": "History",
        "history.revert": "Revert",
        "history.compare": "Compare",
        "history.delete": "Delete",
        "history.info_group": "Information",
        "history.info_none": "Nothing selected",
        # --- SettingsDialog ---
        "settings.window_title": "Settings",
        "settings.gpu_group": "GPU / Performance",
        "settings.cuda_available": "Available",
        "settings.cuda_unavailable": "Unavailable",
        "settings.cuda_na": "N/A",
        "settings.info_text": "PyTorch {ver}  |  CUDA: {status} (driver: {cudaver})",
        "settings.cuda_warning": (
            "CUDA is unavailable. Install PyTorch with CUDA:\n"
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"),
        "settings.device_label": "Compute device:",
        "settings.gpu_option": "GPU {i}: {name} ({mem} GB)",
        "settings.tile_size_label": "Tile size:",
        "settings.tile_size_tooltip": "Larger tiles use more VRAM but process faster",
        "settings.tile_overlap_label": "Tile overlap:",
        "settings.tile_overlap_tooltip": "Overlap between tiles to reduce seam artifacts",
        "settings.gpu_denoise_checkbox": (
            "Replace BM3D with a GPU denoiser (SCUNet) when a GPU is available"),
        "settings.gpu_denoise_tooltip": (
            "BM3D runs on CPU only and is slow. When CUDA is available, the "
            "selected BM3D will be replaced by the fast GPU denoiser SCUNet. "
            "Uncheck to always use BM3D."),
        "settings.output_group": "Output",
        "settings.format_label": "Default format:",
        "settings.output_dir_placeholder": "Same as source",
        "settings.browse": "Browse...",
        "settings.output_dir_label": "Output folder:",
        "settings.history_group": "History",
        "settings.max_entries_label": "Max entries:",
        "settings.retention_label": "Retention:",
        "settings.retention_suffix": " days",
        "settings.storage_group": "Storage",
        "settings.cache_dir_placeholder": "~/.upscaler/models",
        "settings.cache_dir_label": "Model cache:",
        "settings.appearance_group": "Appearance",
        "settings.theme_label": "Theme:",
        "settings.icedit_group": "ICEdit",
        "settings.icedit_offload_label": "Memory offload:",
        "settings.icedit_offload_model": "Model offload (default)",
        "settings.icedit_offload_none": "No offload (high VRAM)",
        "settings.icedit_offload_sequential": "Sequential (minimum VRAM)",
        "settings.icedit_quant_label": "Quantization:",
        "settings.icedit_quant_q4": "Q4 (less VRAM)",
        "settings.icedit_quant_q5": "Q5 (quality)",
        "settings.output_dir_dialog_title": "Select output folder",
        "settings.cache_dir_dialog_title": "Select model cache folder",
        # --- DownloadDialog ---
        "download.title": "Downloading {model}",
        "download.progress_label": "Downloading {model}...",
        "download.cancel": "Cancel",
        # --- CompareDialog ---
        "compare.window_title": "Comparison — Original vs Result",
        "compare.original_label": "Original",
        "compare.result_label": "Result",
        "compare.close_btn": "Close",
        # --- Menus (MainWindow) ---
        "menu.file": "&File",
        "menu.open": "&Open image...",
        "menu.save": "&Save result...",
        "menu.exit": "E&xit",
        "menu.edit": "&Edit",
        "menu.settings": "&Settings...",
        "menu.clear_history": "Clear &history",
        "menu.view": "&View",
        "menu.fit_view": "&Fit to window",
        "menu.processing": "&Processing",
        "menu.process_action": "&Process",
        "menu.cancel_action": "&Cancel",
        "menu.help": "&Help",
        "menu.about": "&About",
        # --- Messages (QMessageBox) ---
        "msg.about_title": "About Upscaler",
        "msg.about_body": (
            "Upscaler — Professional image enhancement\n\n"
            "AI models: Real-ESRGAN, HAT-S, SwinIR, OmniSR, DAT\n"
            "Traditional: Lanczos, Bicubic, Sinc, NEDI, DCCI, EGGI/SIRE\n"
            "Denoising: BM3D, SCUNet, NAFNet, Wavelet, NL-Means, Bilateral\n"
            "Colorization: DDColor, DeOldify, ColorMNet"),
        "msg.resume_session_title": "Resume session",
        "msg.resume_session_body": "Found sessions: {n}. Resume the last one?",
        "msg.error_title": "Error",
        "msg.load_error_body": "Failed to load image:\n{error}",
        "msg.no_image_title": "No image",
        "msg.no_image_body": "Load an image first.",
        "msg.blend_title": "Blend",
        "msg.blend_sources_unavailable": "Not all recipe sources are available.",
        "msg.blend_need_two_variants": (
            "At least two variants are needed (process an image so "
            "pipeline-step variants appear)."),
        "msg.no_instruction_title": "No instruction",
        "msg.no_instruction_body": "Enter an instruction for ICEdit.",
        "msg.result_load_error_body": "Failed to load the result:\n{error}",
        "msg.processing_error_title": "Processing error",
        "msg.processing_error_body": "Stage: {stage}\n\n{error}",
        "msg.no_files_title": "No files",
        "msg.no_files_body": "Add files for batch processing.",
        "msg.clear_history_title": "Clear history",
        "msg.clear_history_body": "Delete all history? This action cannot be undone.",
        # --- File dialogs (MainWindow) ---
        "dialog.open_image_title": "Open image",
        "dialog.images_filter": "Images ({exts});;All files (*)",
        "dialog.open_dir_title": "Select folder with images",
        "dialog.save_result_title": "Save result",
        "dialog.save_filter": "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff);;All files (*)",
        # --- Tooltips (MainWindow) ---
        "tooltip.auto_btn": "Analyze the image and automatically pick settings",
        "tooltip.make_beautiful_btn": "Automatic analysis + parameter selection + processing",
        # --- Face-zones toolbar (MainWindow) ---
        "zones.toolbar_label": "Face zones",
        "zones.toolbar_tooltip": (
            "Face marking mode: draw boxes on the canvas (left click). "
            "Zones replace automatic face detection for CodeFormer."),
        "zones.clear_btn": "Clear",
        # --- Status bar and progress (MainWindow) ---
        "status.ready": "Ready",
        "status.device_cpu_no_torch": "Device: CPU (torch not installed)",
        "status.device_cpu": "Device: CPU",
        "status.device_gpu": "Device: {name}",
        "status.device_cpu_no_cuda": "Device: CPU (CUDA unavailable)",
        "status.disk_gb_warning": "History: {gb:.1f} GB (WARNING)",
        "status.disk_gb": "History: {gb:.1f} GB",
        "status.disk_mb": "History: {mb:.0f} MB",
        "status.loaded": "Loaded: {name} ({w}x{h})",
        "status.llm_refining": "LLM refining processing parameters...",
        "status.auto_applied": "Auto: {desc}",
        "status.blend_preview_ready": "Blend: preview ready",
        "status.blend_saved": "Blend: result saved as version {version}",
        "status.blend_variant_added": "Blend: result added as variant {id}",
        "status.blend_auto_result": "Auto-select: base {base}, layers: {n} (score {score:.3f})",
        "status.auto_processing": "Auto: {desc} — processing...",
        "status.progress": "[{stage}] {message}",
        "status.complete_metrics": "Done | BRISQUE: {brisque:.1f} | NIQE: {niqe:.1f}",
        "status.done": "Done",
        "status.evaluating": "Evaluating result (iteration {iter}/{max})...",
        "status.refining": "Refining (iteration {iter}/{max})...",
        "status.error_stage": "Error: {stage}",
        "status.cancelled": "Cancelled",
        "status.reverted": "Reverted to original",
        "status.batch_done": "Done: {n} files",
        "status.batch_complete": "Batch processing complete",
        "status.colorizing_photo": "Colorizing with {model}...",
        "status.colorizing_video": "Colorizing video with {model}...",
        "status.video_done": "Done: {path}",
        "status.video_error": "Error: {error}",
        "status.saved": "Saved: {path}",
    },
}


class Translator(QObject):
    language_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._lang = "ru"

    def current_language(self) -> str:
        return self._lang

    def set_language(self, lang: str) -> None:
        if lang not in _LANGUAGES or lang == self._lang:
            return
        self._lang = lang
        try:
            from upscaler.config import load_settings, save_settings
            s = load_settings()
            s["language"] = lang
            save_settings(s)
        except Exception:
            pass
        self.language_changed.emit(lang)


_translator = None


def translator() -> Translator:
    global _translator
    if _translator is None:
        _translator = Translator()
        try:
            from upscaler.config import load_settings
            lang = load_settings().get("language", "ru")
            if lang in _LANGUAGES:
                _translator._lang = lang
        except Exception:
            pass
    return _translator


def current_language() -> str:
    return translator().current_language()


def set_language(lang: str) -> None:
    translator().set_language(lang)


def available_languages() -> tuple:
    return _LANGUAGES


def tr(key: str, **fmt) -> str:
    lang = translator().current_language()
    table = _STRINGS.get(lang, {})
    text = table.get(key)
    if text is None:
        other = "en" if lang == "ru" else "ru"
        text = _STRINGS.get(other, {}).get(key, key)
    if fmt:
        try:
            return text.format(**fmt)
        except Exception:
            return text
    return text
