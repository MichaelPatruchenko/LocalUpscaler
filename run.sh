#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

VENV_DIR=".venv"

# Find Python executable (python3 preferred, fallback to python)
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: Python not found. Install Python 3.10+ and add it to PATH."
    exit 1
fi

# Verify Python version >= 3.10
PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PY" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PY" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10+ required, found $PY_VER"
    exit 1
fi
echo "Using Python $PY_VER"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv "$VENV_DIR"
fi

# Activate virtual environment
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"   # Windows (Git Bash / MSYS2)
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"       # Linux / macOS
else
    echo "ERROR: Virtual environment is broken. Delete '$VENV_DIR' and re-run."
    exit 1
fi

# Use python -m pip to avoid Windows pip.exe locking issues
PIP="python -m pip"

# Upgrade pip
echo "Upgrading pip..."
$PIP install --upgrade pip 2>/dev/null || echo "WARNING: Could not upgrade pip, continuing..."

# Install dependencies
echo "Installing dependencies..."
$PIP install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies."
    exit 1
fi

# Run the application
echo "Starting Upscaler..."
python run.py
