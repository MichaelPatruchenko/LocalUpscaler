#!/usr/bin/env bash
# Установка зависимостей Upscaler в .venv на Python 3.10 (Linux/macOS).
#
# Использование:
#   ./install.sh            # CPU-сборки
#   ./install.sh --cuda     # ВСЕ GPU-библиотеки в CUDA-варианте (cu124 по умолчанию):
#                           #   PyTorch, llama-cpp-python, onnxruntime-gpu, стек ICEdit
#   CUDA_VERSION=cu121 ./install.sh --cuda
exec > >(tee -i script.log)
exec 2>&1
set -x  # включение отладки

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
CUDA=0
CUDA_VERSION="${CUDA_VERSION:-cu124}"
for arg in "$@"; do
  [ "$arg" = "--cuda" ] && CUDA=1
done

step() { printf '\n==> %s\n' "$1"; }

# --- 1. Найти Python 3.10 -----------------------------------------------------
step "Поиск Python 3.10"
PY310=""
if command -v python3.10 >/dev/null 2>&1; then
  PY310="python3.10"
elif command -v pyenv >/dev/null 2>&1; then
  echo "Устанавливаю Python 3.10 через pyenv..."
  pyenv install -s 3.10.13
  PY310="$(pyenv root)/versions/3.10.13/bin/python"
else
  echo "Python 3.10 не найден."
  echo "Установите его: Debian/Ubuntu: 'sudo apt install python3.10 python3.10-venv';"
  echo "  macOS: 'brew install python@3.10'; либо поставьте pyenv и перезапустите."
  exit 1
fi
echo "Используется: $PY310 ($($PY310 --version))"

# --- 2. venv ------------------------------------------------------------------
step "Создание .venv"
# Пересоздаём, если venv отсутствует или нерабочий (базовый Python пропал).
venv_ok=0
if [ -x "$VENV_DIR/bin/python" ]; then
  if "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1; then
    venv_ok=1
  else
    echo "Существующий .venv нерабочий — пересоздаю."
  fi
fi
if [ -d "$VENV_DIR" ] && [ "$venv_ok" -eq 0 ]; then
  rm -rf "$VENV_DIR"
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PY310" -m venv "$VENV_DIR"
fi
VPY="$VENV_DIR/bin/python"

# --- 3. pip -------------------------------------------------------------------
step "Обновление pip"
"$VPY" -m pip install --upgrade pip setuptools wheel

# --- 4. PyTorch ---------------------------------------------------------------
step "Установка PyTorch"
if [ "$CUDA" -eq 1 ]; then
  "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
else
  "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
fi

# --- 5. Зависимости -----------------------------------------------------------
step "Установка requirements.txt"
"$VPY" -m pip install -r "$PROJECT_ROOT/requirements.txt"

# --- 6. llama-cpp-python ------------------------------------------------------
step "Установка llama-cpp-python (LLM-советник)"
if [ "$CUDA" -eq 1 ]; then
  LLAMA_INDEX="https://abetlen.github.io/llama-cpp-python/whl/${CUDA_VERSION}"
  # Шаг 1: обычная установка приносит зависимости (и, возможно, CPU-сборку).
  "$VPY" -m pip install --prefer-binary "llama-cpp-python>=0.3" --extra-index-url "$LLAMA_INDEX" || true
  # Шаг 2: принудительно ставим CUDA-колесо, НЕ трогая зависимости. Без
  # --force-reinstall уже стоящая CPU-сборка осталась бы (">=0.3" уже выполнено);
  # --prefer-binary выбирает готовое CUDA-колесо вместо исходника с PyPI.
  "$VPY" -m pip install --upgrade --force-reinstall --no-cache-dir --no-deps --prefer-binary \
    "llama-cpp-python>=0.3" --extra-index-url "$LLAMA_INDEX" || \
    echo "ВНИМАНИЕ: CUDA-сборка llama-cpp-python не установилась — LLM-советник будет работать на CPU/пропускаться."
else
  "$VPY" -m pip install --prefer-binary "llama-cpp-python>=0.3" || \
    echo "ВНИМАНИЕ: llama-cpp-python не установился — приложение работает без LLM-советника."
fi

if [ "$CUDA" -eq 1 ]; then
  # --- 7. onnxruntime-gpu (восстановление лиц) --------------------------------
  step "Установка onnxruntime-gpu (восстановление лиц)"
  "$VPY" -m pip install onnxruntime-gpu || \
    echo "ВНИМАНИЕ: onnxruntime-gpu не установился — восстановление лиц будет пропускаться."

  # --- 8. Стек ICEdit (использует CUDA-torch) --------------------------------
  step "Установка зависимостей ICEdit (diffusers, transformers, accelerate, peft, gguf)"
  "$VPY" -m pip install diffusers transformers accelerate peft gguf sentencepiece protobuf || \
    echo "ВНИМАНИЕ: стек ICEdit не установился — ICEdit будет пропускаться."

  # --- 9. Проверка доступности GPU -------------------------------------------
  step "Проверка доступности GPU"
  "$VPY" - <<'PYEOF' || true
try:
    import torch; print("  PyTorch CUDA:", torch.cuda.is_available())
except Exception as e: print("  PyTorch: не установлен/ошибка:", e)
try:
    import llama_cpp; print("  llama-cpp GPU offload:", llama_cpp.llama_cpp.llama_supports_gpu_offload())
except Exception as e: print("  llama-cpp: не установлен/ошибка:", e)
try:
    import onnxruntime as o
    provs = o.get_available_providers()
    print("  onnxruntime CUDA:", "CUDAExecutionProvider" in provs, provs)
except Exception as e: print("  onnxruntime: не установлен/ошибка:", e)
PYEOF
  echo "Если где-то выше CUDA = False — соответствующий шаг будет работать на CPU."
fi

step "Готово"
echo "Запуск:  $VPY run.py"
echo "Тесты:   $VPY -m pytest tests/ -q"
