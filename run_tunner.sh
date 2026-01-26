#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
PY_MIN="3.9"
VENV_DIR=".venv"
REQ_FILE="requirements.txt"
CFG_FILE="configs/default.yaml"

echo "============================================================"
echo " FoodCLIP Prompt Tuning Runner (macOS/Linux)"
echo "============================================================"

# -----------------------------
# Helpers
# -----------------------------
verlte() { [ "$1" = "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" ]; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

install_python() {
  echo "[!] Python not found. Attempting to install Python..."
  if have_cmd brew; then
    echo "-> Using Homebrew"
    brew update
    brew install python
  elif have_cmd apt-get; then
    echo "-> Using apt-get"
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
  elif have_cmd dnf; then
    echo "-> Using dnf"
    sudo dnf install -y python3 python3-pip
  elif have_cmd yum; then
    echo "-> Using yum"
    sudo yum install -y python3 python3-pip
  else
    echo "[X] No supported package manager found (brew/apt/dnf/yum)."
    echo "    Please install Python 3.9+ manually, then re-run this script."
    exit 1
  fi
}

# -----------------------------
# 1) Check / install Python
# -----------------------------
if have_cmd python3; then
  PY=python3
elif have_cmd python; then
  PY=python
else
  install_python
  if have_cmd python3; then PY=python3; else PY=python; fi
fi

PY_VER="$($PY -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "[i] Detected Python version: $PY_VER"

if ! verlte "$PY_MIN" "$PY_VER"; then
  echo "[X] Python >= $PY_MIN is required. Please upgrade Python and re-run."
  exit 1
fi

# -----------------------------
# 2) Create venv
# -----------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[i] Creating virtual environment: $VENV_DIR"
  $PY -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# -----------------------------
# 3) Install dependencies
# -----------------------------
echo "[i] Upgrading pip..."
python -m pip install --upgrade pip

if [ ! -f "$REQ_FILE" ]; then
  echo "[X] $REQ_FILE not found. Please ensure requirements.txt exists in repo root."
  exit 1
fi

echo "[i] Installing requirements..."
pip install -r "$REQ_FILE"

# -----------------------------
# 4) Sanity checks (folders)
# -----------------------------
mkdir -p results checkpoints

# -----------------------------
# 5) Run pipeline
# -----------------------------
echo "------------------------------------------------------------"
echo "[1/3] Running baseline..."
python -m src.baseline --config "$CFG_FILE" 2>/dev/null || python -m src.baseline
echo "[OK] baseline finished."

echo "------------------------------------------------------------"
echo "[2/3] Running training..."
python -m src.train --config "$CFG_FILE" 2>/dev/null || python -m src.train
echo "[OK] train finished."

echo "------------------------------------------------------------"
echo "[3/3] Running report generation..."
python -m src.report --config "$CFG_FILE" 2>/dev/null || python -m src.report
echo "[OK] report finished."

echo "============================================================"
echo "Done! Outputs saved in:"
echo " - results/"
echo " - checkpoints/"
echo "============================================================"
