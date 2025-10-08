@echo off
set VENV_DIR=venv
if not exist %VENV_DIR% (
    echo [INFO] Creating virtual environment...
    python -m venv %VENV_DIR%
)
echo [INFO] Installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"
pip install -r requirements-2.txt
echo.
echo [SUCCESS] Setup complete.
pause