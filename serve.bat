@echo off
chcp 65001 >nul 2>&1

set MODEL=%~1
set PORT=%~2
if "%PORT%"=="" set PORT=8080

if "%MODEL%"=="" (
    for %%f in ("%~dp0models\*.gguf") do (
        if "%MODEL%"=="" set MODEL=%%f
    )
)
if "%MODEL%"=="" (
    echo.
    echo 오류: 모델 파일을 찾을 수 없습니다.
    echo.
    echo 사용법: serve.bat [모델경로] [포트번호]
    echo 예시:   serve.bat models\Qwen3.5-9B-Q4_K_M.gguf 8080
    echo.
    pause
    exit /b 1
)

if not exist "%~dp0cmake-build-release\mLLM.exe" (
    echo.
    echo 오류: mLLM.exe 가 없습니다. 먼저 setup.bat, build_mllm.bat 을 실행하세요.
    echo.
    pause
    exit /b 1
)

echo.
echo 모델: %MODEL%
echo 포트: %PORT%
echo.
echo API 주소: http://localhost:%PORT%/v1/chat/completions
echo 종료:     Ctrl+C
echo.

set PATH=%~dp0cmake-build-release;%PATH%
"%~dp0cmake-build-release\mLLM.exe" %MODEL% --serve --port %PORT%
