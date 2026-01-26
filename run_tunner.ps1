$ErrorActionPreference = "Stop"

Write-Host "============================================================"
Write-Host " FoodCLIP Prompt Tuning Runner (Windows PowerShell)"
Write-Host "============================================================"

$PY_MIN = [Version]"3.9"
$VENV_DIR = ".venv"
$REQ_FILE = "requirements.txt"
$CFG_FILE = "configs/default.yaml"

function Have-Cmd($name) {
  return (Get-Command $name -ErrorAction SilentlyContinue) -ne $null
}

function Install-Python {
  Write-Host "[!] Python not found. Attempting to install Python..."
  if (Have-Cmd "winget") {
    Write-Host "-> Using winget"
    winget install -e --id Python.Python.3
  } else {
    Write-Host "[X] winget not available. Please install Python 3.9+ manually and re-run."
    exit 1
  }
}

# -----------------------------
# 1) Check / install Python
# -----------------------------
$PY = $null
if (Have-Cmd "python") { $PY = "python" }
elseif (Have-Cmd "py") { $PY = "py" }
else {
  Install-Python
  if (Have-Cmd "python") { $PY = "python" }
  elseif (Have-Cmd "py") { $PY = "py" }
  else {
    Write-Host "[X] Python still not found after install attempt."
    exit 1
  }
}

# Determine version
if ($PY -eq "py") {
  $PY_VER_STR = (& py -3 -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
  $PY_CMD = "py -3"
} else {
  $PY_VER_STR = (& python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
  $PY_CMD = "python"
}

$PY_VER = [Version]$PY_VER_STR
Write-Host "[i] Detected Python version: $PY_VER_STR"

if ($PY_VER -lt $PY_MIN) {
  Write-Host "[X] Python >= $($PY_MIN.ToString()) is required. Please upgrade Python and re-run."
  exit 1
}

# -----------------------------
# 2) Create venv
# -----------------------------
if (-not (Test-Path $VENV_DIR)) {
  Write-Host "[i] Creating virtual environment: $VENV_DIR"
  iex "$PY_CMD -m venv $VENV_DIR"
}

# Activate venv
$activate = Join-Path $VENV_DIR "Scripts\Activate.ps1"
. $activate

# -----------------------------
# 3) Install dependencies
# -----------------------------
Write-Host "[i] Upgrading pip..."
python -m pip install --upgrade pip

if (-not (Test-Path $REQ_FILE)) {
  Write-Host "[X] requirements.txt not found in repo root."
  exit 1
}

Write-Host "[i] Installing requirements..."
pip install -r $REQ_FILE

# -----------------------------
# 4) Create output folders
# -----------------------------
if (-not (Test-Path "results")) { New-Item -ItemType Directory results | Out-Null }
if (-not (Test-Path "checkpoints")) { New-Item -ItemType Directory checkpoints | Out-Null }

# -----------------------------
# 5) Run pipeline
# -----------------------------
Write-Host "------------------------------------------------------------"
Write-Host "[1/3] Running baseline..."
try { python -m src.baseline --config $CFG_FILE } catch { python -m src.baseline }
Write-Host "[OK] baseline finished."

Write-Host "------------------------------------------------------------"
Write-Host "[2/3] Running training..."
try { python -m src.train --config $CFG_FILE } catch { python -m src.train }
Write-Host "[OK] train finished."

Write-Host "------------------------------------------------------------"
Write-Host "[3/3] Running report generation..."
try { python -m src.report --config $CFG_FILE } catch { python -m src.report }
Write-Host "[OK] report finished."

Write-Host "============================================================"
Write-Host "Done! Outputs saved in:"
Write-Host " - results/"
Write-Host " - checkpoints/"
Write-Host "============================================================"
