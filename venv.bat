@echo off
cd /d %~dp0

if not exist .venv (
    echo [.venv not found]
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo.
echo Virtual environment activated.
echo Current path: %cd%
echo.
cmd /k