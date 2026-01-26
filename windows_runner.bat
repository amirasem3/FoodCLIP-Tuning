@echo off
setlocal

echo ============================================================
echo  FoodCLIP Prompt Tuning Runner (Windows BAT)
echo ============================================================

REM -----------------------------
REM 1) Pick Python command (prefer py -3)
REM -----------------------------
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    set "PY="
  )
)

REM -----------------------------
REM 2) If Python missing, try winget install
REM -----------------------------
if "%PY%"=="" (
  echo [!] Python not found. Trying to install with winget...
  where winget >nul 2>nul
  if not %errorlevel%==0 (
    echo [X] winget not available. Install Python 3.9+ manually:
    echo     https://www.python.org/downloads/
    exit /b 1
  )
  winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
  if not %errorlevel%==0 (
    echo [X] Python installation failed.
    exit /b 1
  )

  REM Re-detect after install
  where py >nul 2>nul
  if %errorlevel%==0 (
    set "PY=py -3"
  ) else (
    set "PY=python"
  )
)

REM -----------------------------
REM 3) Basic sanity: does python run?
REM -----------------------------
%PY% -c "import sys; print(sys.version)"
if not %errorlevel%==0 (
  echo [X] Python command failed. If you see Microsoft Store alias warning:
  echo     Disable it:
  echo     Settings -> Apps -> Advanced app settings -> App execution aliases
  echo     Turn OFF python.exe and python3.exe
  exit /b 1
)

REM -----------------------------
REM 4) Create venv
REM -----------------------------
if not exist .venv (
  echo [i] Creating virtual environment...
  %PY% -m venv .venv
  if not %errorlevel%==0 exit /b 1
)

REM -----------------------------
REM 5) Activate venv
REM -----------------------------
call .venv\Scripts\activate.bat

REM -----------------------------
REM 6) Install requirements
REM -----------------------------
if not exist requirements.txt (
  echo [X] requirements.txt not found in repo root.
  exit /b 1
)

echo [i] Upgrading pip...
python -m pip install --upgrade pip

echo [i] Installing requirements...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch
pip install -r requirements.txt

if not %errorlevel%==0 exit /b 1

REM -----------------------------
REM 7) Ensure folders exist
REM -----------------------------
if not exist results mkdir results
if not exist checkpoints mkdir checkpoints

REM -----------------------------
REM 8) Run pipeline
REM -----------------------------
echo ------------------------------------------------------------
echo [1/3] Running baseline...
python -m src.baseline
if not %errorlevel%==0 exit /b 1

echo ------------------------------------------------------------
echo [2/3] Running training...
python -m src.train
if not %errorlevel%==0 exit /b 1

echo ------------------------------------------------------------
echo [3/3] Running report...
python -m src.report
if not %errorlevel%==0 exit /b 1

echo ============================================================
echo Done! Outputs saved in results\ and checkpoints\
echo ============================================================

endlocal