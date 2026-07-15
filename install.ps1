<#
.SYNOPSIS
    Установка всех зависимостей Upscaler в изолированное окружение Python 3.10.

.DESCRIPTION
    Скрипт:
      1. Находит Python 3.10 (или устанавливает его через winget).
      2. Создаёт виртуальное окружение .venv на базе Python 3.10.
      3. Ставит PyTorch (CPU по умолчанию или CUDA с -Cuda).
      4. Ставит зависимости из requirements.txt.
      5. Ставит llama-cpp-python для локального LLM-советника (CPU или CUDA).
      6. С -Cuda также ставит onnxruntime-gpu (восстановление лиц) и стек
         ICEdit (diffusers и пр.), и проверяет, что GPU реально доступен.

.PARAMETER Cuda
    Установить ВСЕ работающие с GPU библиотеки в CUDA-вариантах: PyTorch,
    llama-cpp-python (LLM/vision-советник), onnxruntime-gpu (CodeFormer) и стек
    ICEdit (diffusers, transformers, accelerate, peft, gguf).

.PARAMETER CudaVersion
    Тег CUDA для индексов колёс (по умолчанию cu124 — валиден и для индекса
    PyTorch, и для индекса llama-cpp-python).

.PARAMETER Recreate
    Пересоздать .venv, если он уже существует.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Cuda
    .\install.ps1 -Cuda -CudaVersion cu124 -Recreate
#>
[CmdletBinding()]
param(
    [switch]$Cuda,
    [string]$CudaVersion = "cu124",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --- 1. Найти/установить Python 3.10 ------------------------------------------
Write-Step "Поиск Python 3.10"
$py310 = $null
try {
    $ver = & py -3.10 --version 2>$null
    if ($ver) { $py310 = "py -3.10" }
} catch {}

if (-not $py310) {
    Write-Host "Python 3.10 не найден. Пытаюсь установить через winget..." -ForegroundColor Yellow
    $winget = (Get-Command winget -ErrorAction SilentlyContinue)
    if ($null -eq $winget) {
        Write-Error "winget недоступен. Установите Python 3.10 вручную с https://www.python.org/downloads/release/python-31011/ и перезапустите скрипт."
        exit 1
    }
    winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
    Write-Host "Python 3.10 установлен. Возможно, потребуется перезапустить терминал, если 'py -3.10' не виден." -ForegroundColor Yellow
    $py310 = "py -3.10"
}
Write-Host "Используется интерпретатор: $py310"

# --- 2. Создать venv ----------------------------------------------------------
Write-Step "Создание виртуального окружения .venv"

# Проверяем, что существующий venv рабочий (его базовый интерпретатор на месте).
$venvOk = $false
if ((Test-Path $VenvPython) -and (-not $Recreate)) {
    try {
        & $VenvPython -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $venvOk = $true }
    } catch {}
    if (-not $venvOk) {
        Write-Host "Существующий .venv нерабочий (базовый Python отсутствует) — пересоздаю." -ForegroundColor Yellow
    }
}

if ((Test-Path $VenvDir) -and ($Recreate -or (-not $venvOk))) {
    Write-Host "Удаляю .venv"
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    & py -3.10 -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Не удалось создать venv"; exit 1 }
} else {
    Write-Host ".venv рабочий — переиспользую (используйте -Recreate для пересоздания)"
}

# --- 3. Обновить pip ----------------------------------------------------------
Write-Step "Обновление pip/setuptools/wheel"
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { Write-Error "Не удалось обновить pip"; exit 1 }

# --- 4. PyTorch ---------------------------------------------------------------
Write-Step "Установка PyTorch"
if ($Cuda) {
    Write-Host "Сборка CUDA ($CudaVersion)"
    & $VenvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaVersion"
} else {
    Write-Host "Сборка CPU"
    & $VenvPython -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
}
if ($LASTEXITCODE -ne 0) { Write-Error "Не удалось установить PyTorch"; exit 1 }

# --- 5. Зависимости проекта ---------------------------------------------------
Write-Step "Установка зависимостей из requirements.txt"
$req = Join-Path $ProjectRoot "requirements.txt"
& $VenvPython -m pip install -r $req
if ($LASTEXITCODE -ne 0) { Write-Error "Не удалось установить зависимости из requirements.txt"; exit 1 }

# --- 6. llama-cpp-python (LLM-советник) ---------------------------------------
Write-Step "Установка llama-cpp-python (локальный LLM-советник)"
if ($Cuda) {
    $LlamaIndex = "https://abetlen.github.io/llama-cpp-python/whl/$CudaVersion"
    # Шаг 1: обычная установка приносит зависимости (и, возможно, CPU-сборку).
    & $VenvPython -m pip install --prefer-binary "llama-cpp-python>=0.3" --extra-index-url $LlamaIndex
    # Шаг 2: принудительно ставим CUDA-колесо, НЕ трогая зависимости (--no-deps).
    # Это критично: без --force-reinstall уже стоящая CPU-сборка осталась бы
    # (требование ">=0.3" уже выполнено и pip ничего бы не делал), а
    # --prefer-binary выбирает готовое CUDA-колесо вместо исходника с PyPI
    # (который собрался бы в CPU-режиме или потребовал бы компилятор).
    & $VenvPython -m pip install --upgrade --force-reinstall --no-cache-dir --no-deps --prefer-binary "llama-cpp-python>=0.3" --extra-index-url $LlamaIndex
} else {
    & $VenvPython -m pip install --prefer-binary "llama-cpp-python>=0.3"
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "ВНИМАНИЕ: llama-cpp-python не установился (нет колеса под вашу платформу/CUDA-тег или требуется компилятор)." -ForegroundColor Yellow
    Write-Host "Приложение будет работать без LLM-советника (откат на алгоритмический режим)." -ForegroundColor Yellow
}

if ($Cuda) {
    # --- 7. onnxruntime-gpu (восстановление лиц CodeFormer) -------------------
    Write-Step "Установка onnxruntime-gpu (восстановление лиц)"
    & $VenvPython -m pip install onnxruntime-gpu
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ВНИМАНИЕ: onnxruntime-gpu не установился — восстановление лиц будет пропускаться." -ForegroundColor Yellow
    }

    # --- 8. Стек ICEdit (FLUX.1-Fill-dev, использует CUDA-torch) --------------
    Write-Step "Установка зависимостей ICEdit (diffusers, transformers, accelerate, peft, gguf)"
    & $VenvPython -m pip install diffusers transformers accelerate peft gguf sentencepiece protobuf
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ВНИМАНИЕ: стек ICEdit не установился — ICEdit будет пропускаться." -ForegroundColor Yellow
    }

    # --- 9. Проверка, что GPU реально доступен --------------------------------
    Write-Step "Проверка доступности GPU"
    $GpuCheck = @'
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
'@
    & $VenvPython -c $GpuCheck
    Write-Host "Если где-то выше CUDA = False — соответствующий шаг будет работать на CPU." -ForegroundColor Yellow
}

# --- Готово -------------------------------------------------------------------
Write-Step "Готово"
Write-Host "Запуск приложения:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\python.exe run.py" -ForegroundColor Green
Write-Host "Запуск тестов:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\python.exe -m pytest tests\ -q" -ForegroundColor Green
