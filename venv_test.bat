@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

REM ----------------------------
REM .venv 생성
REM ----------------------------
if not exist .venv (
    echo [.venv not found]
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM ----------------------------
REM scripts 폴더의 .py 파일 목록 출력
REM ----------------------------
echo.
echo ===== Available Python Scripts =====

set count=0

for %%f in (scripts\*.py) do (
    set /a count+=1
    set file!count!=%%~nxf
    echo !count!. %%~nxf
)

if %count%==0 (
    echo No python scripts found in scripts folder.
    pause
    exit /b
)

echo.
set /p choice=Select script number to run: 

REM ----------------------------
REM 입력 검증
REM ----------------------------
if not defined file%choice% (
    echo Invalid selection.
    pause
    exit /b
)

set selected=!file%choice%!

echo.
echo Running: %selected%
echo.

python scripts\%selected%

echo.
echo Virtual environment activated.
echo Current path: %cd%
echo.
cmd /k