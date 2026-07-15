"""Automated build script for Upscaler application.

Usage:
    python build.py              # Full build (folder mode, with CUDA if available)
    python build.py --onefile    # Single .exe file
    python build.py --cpu        # CPU-only build (smaller)
    python build.py --no-models  # Build without bundling AI models
    python build.py --clean      # Clean build artifacts before building

Requirements:
    pip install pyinstaller
    pip install -r requirements.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "upscaler" / "models" / "models"
ARCH_DIR = ROOT_DIR / "upscaler" / "models" / "arch"
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"
SPEC_FILE = ROOT_DIR / "Upscaler.spec"
ENTRY_POINT = ROOT_DIR / "run.py"

# ── Plugin discovery ───────────────────────────────────────────────────────

PLUGIN_DIRS = [
    ROOT_DIR / "upscaler" / "plugins" / "upscalers",
    ROOT_DIR / "upscaler" / "plugins" / "denoisers",
    ROOT_DIR / "upscaler" / "plugins" / "adjusters",
    ROOT_DIR / "upscaler" / "plugins" / "colorizers",
]

ARCH_MODULES = [
    "upscaler.models.arch.ddcolor_arch",
    "upscaler.models.arch.deoldify_arch",
]

EXTRA_HIDDEN = [
    "spandrel",
    "safetensors",
    "einops",
]


def discover_plugins() -> list[str]:
    """Find all plugin modules automatically."""
    plugins = []
    for plugin_dir in PLUGIN_DIRS:
        if not plugin_dir.exists():
            continue
        for py_file in sorted(plugin_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            # Convert file path to module path
            rel = py_file.relative_to(ROOT_DIR)
            module = str(rel.with_suffix("")).replace(os.sep, ".")
            plugins.append(module)
    return plugins


def check_prerequisites():
    """Verify that required tools and files are present."""
    errors = []

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__}")
    except ImportError:
        errors.append("PyInstaller не установлен. Установите: pip install pyinstaller")

    # Check torch
    try:
        import torch
        cuda_status = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "CPU only"
        print(f"  PyTorch {torch.__version__} ({cuda_status})")
    except ImportError:
        errors.append("PyTorch не установлен. См. requirements.txt")

    # Check PySide6
    try:
        import PySide6
        print(f"  PySide6 {PySide6.__version__}")
    except ImportError:
        errors.append("PySide6 не установлен. Установите: pip install PySide6")

    # Check spandrel
    try:
        import spandrel
        print(f"  spandrel {spandrel.__version__}")
    except ImportError:
        errors.append("spandrel не установлен. Установите: pip install spandrel")

    # Check entry point
    if not ENTRY_POINT.exists():
        errors.append(f"Точка входа не найдена: {ENTRY_POINT}")

    if errors:
        print("\nОшибки:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)


def check_models():
    """List available models."""
    if not MODELS_DIR.exists():
        print(f"\n  Папка моделей не найдена: {MODELS_DIR}")
        return []

    model_files = sorted(MODELS_DIR.glob("*"))
    model_files = [f for f in model_files if f.is_file() and not f.name.startswith(".")]

    total_mb = sum(f.stat().st_size for f in model_files) / (1024 * 1024)
    print(f"\n  Найдено моделей: {len(model_files)} ({total_mb:.0f} МБ)")
    for f in model_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size_mb:.1f} МБ)")

    return model_files


def clean_build():
    """Remove previous build artifacts."""
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            print(f"  Удаление {d}")
            shutil.rmtree(d)
    # Remove generated .spec (we generate our own)
    for spec in ROOT_DIR.glob("*.spec"):
        if spec.name != "Upscaler.spec":
            spec.unlink()


def build(onefile: bool = False, include_models: bool = True):
    """Run PyInstaller build."""
    # Discover all plugins
    plugins = discover_plugins()
    hidden_imports = plugins + ARCH_MODULES + EXTRA_HIDDEN

    print(f"\n  Обнаружено плагинов: {len(plugins)}")
    for p in plugins:
        print(f"    {p}")

    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "Upscaler",
        "--windowed",
        "--noconfirm",
    ]

    if onefile:
        cmd.append("--onefile")

    # Hidden imports
    for hi in hidden_imports:
        cmd.extend(["--hidden-import", hi])

    # Collect all for packages that use dynamic loading
    for pkg in ("spandrel", "torch"):
        cmd.extend(["--collect-all", pkg])

    # Add model files as data
    if include_models and MODELS_DIR.exists():
        # Use os.pathsep-based format: source;dest on Windows, source:dest on Unix
        sep = ";" if sys.platform == "win32" else ":"
        cmd.extend(["--add-data", f"{MODELS_DIR}{sep}upscaler/models/models"])

    # Add architecture files (ddcolor_arch, deoldify_arch)
    if ARCH_DIR.exists():
        sep = ";" if sys.platform == "win32" else ":"
        cmd.extend(["--add-data", f"{ARCH_DIR}{sep}upscaler/models/arch"])

    # Exclude unused heavy modules
    for exclude in ("torch.utils.tensorboard", "tensorboard", "matplotlib", "tkinter"):
        cmd.extend(["--exclude-module", exclude])

    # Entry point
    cmd.append(str(ENTRY_POINT))

    print(f"\n  Команда сборки:")
    # Print shortened version for readability
    print(f"    pyinstaller {'--onefile ' if onefile else ''}--name Upscaler ...")
    print(f"    ({len(hidden_imports)} hidden imports, models={'yes' if include_models else 'no'})")

    print("\n═══ Запуск PyInstaller ═══\n")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))

    if result.returncode != 0:
        print(f"\n✗ Сборка завершилась с ошибкой (код {result.returncode})")
        sys.exit(result.returncode)

    # Report result
    if onefile:
        exe_path = DIST_DIR / "Upscaler.exe"
    else:
        exe_path = DIST_DIR / "Upscaler" / "Upscaler.exe"

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\n✓ Сборка завершена: {exe_path} ({size_mb:.1f} МБ)")

        if not onefile:
            dist_folder = DIST_DIR / "Upscaler"
            total = sum(
                f.stat().st_size for f in dist_folder.rglob("*") if f.is_file()
            ) / (1024 * 1024)
            print(f"  Общий размер папки: {total:.0f} МБ")
    else:
        print(f"\n✗ Исполняемый файл не найден: {exe_path}")
        sys.exit(1)


def run_tests():
    """Run test suite before building."""
    print("\n═══ Запуск тестов ═══\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
        cwd=str(ROOT_DIR),
    )
    if result.returncode != 0:
        print("\n✗ Тесты не прошли. Исправьте ошибки перед сборкой.")
        sys.exit(result.returncode)
    print()


def main():
    parser = argparse.ArgumentParser(description="Сборка Upscaler в исполняемый файл")
    parser.add_argument("--onefile", action="store_true",
                        help="Собрать в один .exe файл (медленный запуск)")
    parser.add_argument("--cpu", action="store_true",
                        help="CPU-only сборка (без проверки CUDA)")
    parser.add_argument("--no-models", action="store_true",
                        help="Не включать модели в сборку")
    parser.add_argument("--no-tests", action="store_true",
                        help="Пропустить тесты перед сборкой")
    parser.add_argument("--clean", action="store_true",
                        help="Очистить артефакты предыдущей сборки")
    args = parser.parse_args()

    print("═══ Upscaler Build ═══\n")

    # Step 1: Check prerequisites
    print("1. Проверка зависимостей:")
    check_prerequisites()

    # Step 2: Check models
    print("\n2. Модели:")
    if args.no_models:
        print("  Пропущено (--no-models)")
    else:
        models = check_models()
        if not models:
            print("  ⚠ Модели не найдены. Используйте --no-models или скачайте модели.")
            print(f"    Путь: {MODELS_DIR}")

    # Step 3: Clean
    if args.clean:
        print("\n3. Очистка:")
        clean_build()
    else:
        print("\n3. Очистка: пропущено (используйте --clean)")

    # Step 4: Tests
    if not args.no_tests:
        print("\n4. Тесты:")
        run_tests()
    else:
        print("\n4. Тесты: пропущено (--no-tests)")

    # Step 5: Build
    print("5. Сборка:")
    build(
        onefile=args.onefile,
        include_models=not args.no_models,
    )

    print("\n═══ Готово ═══")


if __name__ == "__main__":
    main()
