@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create venv. Make sure Python 3.10+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo Installing dependencies...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

set "PYTHONPATH=%~dp0"
".venv\Scripts\python.exe" -m better_cro_lidar.desktop %*
