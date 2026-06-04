@echo off
chcp 65001 >nul 2>&1
echo.
echo === mLLM 환경 설정 ===
echo.
echo configure.ps1 을 실행합니다...
powershell -ExecutionPolicy Bypass -File "%~dp0configure.ps1"
echo.
pause
