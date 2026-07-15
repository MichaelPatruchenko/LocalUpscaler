# Сборка Upscaler в .exe (Windows)

Пошаговая инструкция по сборке приложения в исполняемый файл на Windows.

## 1. Подготовка окружения

### Системные требования

- Windows 10/11 (x64)
- Python 3.10+ (рекомендуется 3.10 — совместим со всеми зависимостями)
- ~15 ГБ свободного места на диске (Python + зависимости + модели + сборка)
- Microsoft Visual C++ Redistributable 2015–2022: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Установка Python

Скачайте Python 3.10 с https://www.python.org/downloads/ и при установке отметьте **"Add Python to PATH"**.

Проверка:

```cmd
python --version
pip --version
```

### Создание виртуального окружения (рекомендуется)

```cmd
cd path\to\Upscaler
python -m venv .venv
.venv\Scripts\activate
```

## 2. Установка зависимостей

### Вариант A: С GPU (CUDA) — рекомендуется

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pyinstaller
```

> Для CUDA нужна совместимая видеокарта NVIDIA и установленные драйверы. CUDA toolkit устанавливать отдельно НЕ нужно — PyTorch включает его.

### Вариант B: Без GPU (CPU-only) — лёгкая сборка

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pyinstaller
```

> CPU-сборка: ~300-400 МБ вместо 2-3 ГБ, но обработка ИИ-моделями значительно медленнее.

### Проверка установки

```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import PySide6; print(f'PySide6 OK')"
python -c "import spandrel; print(f'Spandrel OK')"
```

## 3. Загрузка моделей

Все ИИ-модели хранятся в `upscaler/models/models/`. При первом использовании модели с URL она скачивается автоматически в эту папку. Для сборки .exe все нужные модели должны быть уже скачаны.

### Быстрый способ — запустить приложение и обработать изображение каждой моделью

```cmd
python run.py
```

Выберите каждый ИИ-апскейлер и обработайте любое изображение — модель скачается автоматически.

### Ручная загрузка всех моделей скриптом

```cmd
python -c "
from upscaler.models.manager import ModelManager, MODEL_REGISTRY, LOCAL_MODELS_DIR
from pathlib import Path
mm = ModelManager(cache_dir=Path.home() / '.upscaler' / 'models')
for name, info in MODEL_REGISTRY.items():
    if info.get('url'):
        if not mm.is_downloaded(name):
            print(f'Загрузка {name} ({info[\"size_mb\"]} МБ)...')
            mm.download(name, progress_cb=lambda p,d,t: print(f'  {p}%%', end='\r'))
            print(f'  {name} — готово')
        else:
            print(f'{name} — уже загружена')
    else:
        path = mm.get_model_path(name)
        status = 'найдена' if path.exists() else 'ОТСУТСТВУЕТ'
        print(f'{name} — {status} ({path})')
"
```

### Полный список моделей

| Файл | Модель | Размер | Источник |
|------|--------|--------|----------|
| `RealESRGAN_x4plus.pth` | Real-ESRGAN x4 | 65 МБ | Авто-загрузка |
| `RealESRGAN_x2plus.pth` | Real-ESRGAN x2 | 65 МБ | Авто-загрузка |
| `SwinIR_x2.pth` | SwinIR x2 | 50 МБ | Авто-загрузка |
| `SwinIR_x4.pth` | SwinIR x4 | 50 МБ | Авто-загрузка |
| `HAT-S_x4.safetensors` | HAT-S x4 | 40 МБ | Авто-загрузка |
| `OmniSR_X2_DIV2K.safetensors` | OmniSR x2 | 2 МБ | Авто-загрузка |
| `OmniSR_X4_DIV2K.safetensors` | OmniSR x4 | 2 МБ | Авто-загрузка |
| `DAT_x2.pth` | DAT x2 | 45 МБ | Авто-загрузка |
| `4xNomos8kDAT.safetensors` | DAT x4 | 154 МБ | Авто-загрузка |
| `scunet_color_real_psnr.pth` | SCUNet | 25 МБ | Авто-загрузка |
| `ddcolor_artistic.pth` | DDColor Artistic | 912 МБ | Авто-загрузка |
| `ddcolor_modelscope.pth` | DDColor ModelScope | 912 МБ | Авто-загрузка |
| `NAFNet-SIDD-width64.pth` | NAFNet (SIDD) | 464 МБ | Ручная установка |
| `NAFNet-GoPro-width64.pth` | NAFNet (GoPro) | 272 МБ | Ручная установка |
| `ColorizeStable_gen.pth` | DeOldify Stable | 874 МБ | Ручная установка |
| `ColorizeArtistic_gen.pth` | DeOldify Artistic | 255 МБ | Ручная установка |
| `ColorizeVideo_gen.pth` | DeOldify Video | 874 МБ | Ручная установка |
| `DINOv2FeatureV6_LocalAtten_s2_154000.pth` | ColorMNet | 495 МБ | Ручная установка |

> **Ручная установка** — у этих моделей нет публичного URL для автозагрузки. Скачайте их вручную и поместите в `upscaler/models/models/`.

## 4. Проверка перед сборкой

Убедитесь, что приложение работает из исходников:

```cmd
python run.py
```

Запустите тесты:

```cmd
python -m pytest tests/ -v
```

## 5. Сборка

### Вариант 1: Папка с файлами (рекомендуется)

Быстрый запуск, удобно для тестирования. Результат: папка `dist/Upscaler/`.

```cmd
pyinstaller --windowed --name Upscaler ^
    --hidden-import=upscaler.plugins.upscalers.real_esrgan ^
    --hidden-import=upscaler.plugins.upscalers.hat_s ^
    --hidden-import=upscaler.plugins.upscalers.swinir ^
    --hidden-import=upscaler.plugins.upscalers.omnisr ^
    --hidden-import=upscaler.plugins.upscalers.dat ^
    --hidden-import=upscaler.plugins.upscalers.lanczos ^
    --hidden-import=upscaler.plugins.upscalers.bicubic ^
    --hidden-import=upscaler.plugins.upscalers.sinc ^
    --hidden-import=upscaler.plugins.upscalers.nedi ^
    --hidden-import=upscaler.plugins.upscalers.dcci ^
    --hidden-import=upscaler.plugins.upscalers.eggi_sire ^
    --hidden-import=upscaler.plugins.denoisers.scunet ^
    --hidden-import=upscaler.plugins.denoisers.nafnet ^
    --hidden-import=upscaler.plugins.denoisers.bm3d_plugin ^
    --hidden-import=upscaler.plugins.denoisers.nl_means ^
    --hidden-import=upscaler.plugins.denoisers.bilateral ^
    --hidden-import=upscaler.plugins.denoisers.wavelet ^
    --hidden-import=upscaler.plugins.adjusters.auto_tone ^
    --hidden-import=upscaler.plugins.adjusters.auto_contrast ^
    --hidden-import=upscaler.plugins.adjusters.auto_color ^
    --hidden-import=upscaler.plugins.adjusters.brightness ^
    --hidden-import=upscaler.plugins.adjusters.contrast ^
    --hidden-import=upscaler.plugins.adjusters.saturation ^
    --hidden-import=upscaler.plugins.adjusters.sharpness ^
    --hidden-import=upscaler.plugins.adjusters.refocus ^
    --hidden-import=upscaler.plugins.colorizers.ddcolor ^
    --hidden-import=upscaler.plugins.colorizers.deoldify ^
    --hidden-import=upscaler.plugins.colorizers.colormnet ^
    --hidden-import=upscaler.models.arch.ddcolor_arch ^
    --hidden-import=upscaler.models.arch.deoldify_arch ^
    --hidden-import=spandrel ^
    --hidden-import=safetensors ^
    --hidden-import=einops ^
    --collect-all spandrel ^
    --collect-all torch ^
    --add-data "upscaler/models/models;upscaler/models/models" ^
    --exclude-module matplotlib ^
    --exclude-module tkinter ^
    --exclude-module unittest ^
    run.py
```

Запуск:

```cmd
dist\Upscaler\Upscaler.exe
```

### Вариант 2: Один .exe файл

Проще распространять, но каждый запуск распаковывает файлы во временную папку (~30-60 секунд).

Добавьте флаг `--onefile` к команде выше:

```cmd
pyinstaller --onefile --windowed --name Upscaler ^
    --hidden-import=upscaler.plugins.upscalers.real_esrgan ^
    --hidden-import=upscaler.plugins.upscalers.hat_s ^
    --hidden-import=upscaler.plugins.upscalers.swinir ^
    --hidden-import=upscaler.plugins.upscalers.omnisr ^
    --hidden-import=upscaler.plugins.upscalers.dat ^
    --hidden-import=upscaler.plugins.upscalers.lanczos ^
    --hidden-import=upscaler.plugins.upscalers.bicubic ^
    --hidden-import=upscaler.plugins.upscalers.sinc ^
    --hidden-import=upscaler.plugins.upscalers.nedi ^
    --hidden-import=upscaler.plugins.upscalers.dcci ^
    --hidden-import=upscaler.plugins.upscalers.eggi_sire ^
    --hidden-import=upscaler.plugins.denoisers.scunet ^
    --hidden-import=upscaler.plugins.denoisers.nafnet ^
    --hidden-import=upscaler.plugins.denoisers.bm3d_plugin ^
    --hidden-import=upscaler.plugins.denoisers.nl_means ^
    --hidden-import=upscaler.plugins.denoisers.bilateral ^
    --hidden-import=upscaler.plugins.denoisers.wavelet ^
    --hidden-import=upscaler.plugins.adjusters.auto_tone ^
    --hidden-import=upscaler.plugins.adjusters.auto_contrast ^
    --hidden-import=upscaler.plugins.adjusters.auto_color ^
    --hidden-import=upscaler.plugins.adjusters.brightness ^
    --hidden-import=upscaler.plugins.adjusters.contrast ^
    --hidden-import=upscaler.plugins.adjusters.saturation ^
    --hidden-import=upscaler.plugins.adjusters.sharpness ^
    --hidden-import=upscaler.plugins.adjusters.refocus ^
    --hidden-import=upscaler.plugins.colorizers.ddcolor ^
    --hidden-import=upscaler.plugins.colorizers.deoldify ^
    --hidden-import=upscaler.plugins.colorizers.colormnet ^
    --hidden-import=upscaler.models.arch.ddcolor_arch ^
    --hidden-import=upscaler.models.arch.deoldify_arch ^
    --hidden-import=spandrel ^
    --hidden-import=safetensors ^
    --hidden-import=einops ^
    --collect-all spandrel ^
    --collect-all torch ^
    --add-data "upscaler/models/models;upscaler/models/models" ^
    --exclude-module matplotlib ^
    --exclude-module tkinter ^
    --exclude-module unittest ^
    run.py
```

Результат: `dist/Upscaler.exe` — один файл (~2-3 ГБ с CUDA, ~300-400 МБ без CUDA).

## 6. Проверка сборки

```cmd
:: Запуск с консолью для диагностики (если --windowed скрывает ошибки)
dist\Upscaler\Upscaler.exe

:: Или пересоберите без --windowed для отладки:
pyinstaller --name Upscaler --console ^
    ... (те же флаги) ...
    run.py
```

Что проверить:
1. Приложение запускается без ошибок
2. Все плагины видны в списках (11 апскейлеров, 6 шумоподавителей, 7 корректоров, 3 колоризатора)
3. Обработка изображения работает (Lanczos — без модели, быстрая проверка)
4. ИИ-модели загружаются (Real-ESRGAN — с моделью)

## 7. Размер сборки

| Конфигурация | Размер dist/ | Размер с моделями |
|-------------|-------------|-------------------|
| CUDA (cu121) | ~2.0 ГБ | ~5.5 ГБ |
| CPU-only | ~300 МБ | ~3.8 ГБ |

Модели (`upscaler/models/models/`) занимают ~3.5 ГБ (все модели).

### Сборка без тяжёлых моделей

Если не нужны все модели, удалите ненужные из `upscaler/models/models/` перед сборкой. Минимальный набор для базовой работы:

```
RealESRGAN_x4plus.pth       65 МБ   (основной апскейлер)
RealESRGAN_x2plus.pth       65 МБ   (для 2x)
scunet_color_real_psnr.pth  25 МБ   (шумоподавитель)
```

## 8. Структура файлов после сборки

```
dist/Upscaler/
├── Upscaler.exe                        # Точка входа
├── upscaler/models/models/             # ИИ-модели
│   ├── RealESRGAN_x4plus.pth
│   ├── SwinIR_x4.pth
│   ├── ddcolor_artistic.pth
│   └── ...
├── torch/                              # PyTorch + CUDA DLL
├── PySide6/                            # Qt6
├── cv2/                                # OpenCV
├── spandrel/                           # Архитектуры SR-моделей
└── ...                                 # Остальные зависимости
```

## Возможные проблемы

### PyTorch CUDA не работает в собранном .exe

Убедитесь, что `--collect-all torch` указан. PyTorch хранит CUDA-библиотеки в подпапках, которые PyInstaller может пропустить.

### Модели SR не загружаются (spandrel)

Обязательно добавьте `--collect-all spandrel` — эта библиотека использует динамическую загрузку архитектур.

### Ошибка "DLL not found"

Установите Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Плагины не обнаруживаются

Все плагины перечислены в `--hidden-import`. Если вы добавили новый плагин, добавьте его в список.

### Архитектуры DDColor/DeOldify не найдены

Убедитесь, что `--hidden-import=upscaler.models.arch.ddcolor_arch` и `--hidden-import=upscaler.models.arch.deoldify_arch` указаны.

### Локальные модели не найдены в .exe

Убедитесь, что `--add-data "upscaler/models/models;upscaler/models/models"` указан в команде сборки.

### ColorMNet: ошибка загрузки DINOv2

ColorMNet извлекает DINOv2 ViT-S/14 из своего чекпоинта (`DINOv2FeatureV6_LocalAtten_s2_154000.pth`). При первом запуске `torch.hub` загружает определение архитектуры DINOv2 из GitHub (~100 КБ, не веса). Для оффлайн-работы заранее запустите:

```cmd
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False)"
```

Это закэширует определение архитектуры в `%USERPROFILE%\.cache\torch\hub\`.

### Ошибка "No module named ..."

PyInstaller может пропустить зависимости. Добавьте их через `--hidden-import=имя_модуля`.

### Антивирус блокирует .exe

PyInstaller-сборки часто вызывают ложные срабатывания антивирусов. Добавьте `dist/Upscaler/` в исключения антивируса или подпишите .exe цифровой подписью.

### Долгая сборка / нехватка памяти

`--collect-all torch` копирует весь PyTorch (~2 ГБ). Это нормально. Убедитесь, что на диске достаточно места (15+ ГБ).
