@echo off
REM ============================================================================
REM  Nairobi Urban Flood Digital Twin — Bootstrap Script (Windows)
REM  One-command setup: venv creation → pip upgrade → install → scaffold → verify
REM
REM  Usage:   bootstrap.bat
REM  Expects: Python 3.10+ on PATH
REM ============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   Nairobi Flood Digital Twin — Full Bootstrap
echo ============================================================
echo.

REM ---------- 1. Verify Python version ----------
echo [1/6] Checking Python version ...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python is not on PATH. Install Python 3.10+ and retry.
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
    if %%a LSS 3 (
        echo ERROR: Python 3.10+ required, found !PYVER!
        exit /b 1
    )
    if %%a==3 if %%b LSS 10 (
        echo ERROR: Python 3.10+ required, found !PYVER!
        exit /b 1
    )
)
echo   Found Python !PYVER! — OK

REM ---------- 2. Create virtual environment ----------
echo.
echo [2/6] Creating virtual environment (.venv) ...
if not exist ".venv" (
    python -m venv .venv
    echo   Virtual environment created.
) else (
    echo   .venv already exists — skipping creation.
)

REM ---------- 3. Activate venv ----------
echo.
echo [3/6] Activating virtual environment ...
call .venv\Scripts\activate.bat

REM ---------- 4. Upgrade native tools ----------
echo.
echo [4/6] Upgrading pip, setuptools, wheel ...
python -m pip install --upgrade pip setuptools wheel --quiet

REM ---------- 5. Install requirements ----------
echo.
echo [5/6] Installing pinned requirements (this may take a few minutes) ...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. Check the output above for details.
    exit /b 1
)
echo   All dependencies installed successfully.

REM ---------- 6. Run workspace scaffold ----------
echo.
echo [6/6] Running workspace layout generator ...
call setup_workspace.bat

REM ---------- 7. Verify memory constraints ----------
echo.
echo ============================================================
echo   Running memory diagnostics ...
echo ============================================================
python -m src.utils.memory_check

echo.
echo ============================================================
echo   Bootstrap complete!
echo ============================================================
echo.
echo   To activate the environment in future sessions:
echo     .venv\Scripts\activate.bat
echo.
echo   To launch the dashboard (when implemented):
echo     python -m src.dashboard.app
echo.

endlocal
