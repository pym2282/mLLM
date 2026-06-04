@echo off
chcp 65001 >nul 2>&1

:: 모델 경로: 인수로 지정하거나 models\ 폴더에서 자동 감지
set MODEL=%~1
if "%MODEL%"=="" (
    for %%f in ("%~dp0models\*.gguf") do (
        if "%MODEL%"=="" set MODEL=%%f
    )
)
if "%MODEL%"=="" (
    echo.
    echo 오류: 모델 파일을 찾을 수 없습니다.
    echo.
    echo 사용법: chat.bat [모델경로]
    echo 예시:   chat.bat models\Qwen3.5-9B-Q4_K_M.gguf
    echo.
    echo 모델 다운로드: python scripts\download_model.py --help
    echo.
    pause
    exit /b 1
)

:: 빌드 확인
if not exist "%~dp0cmake-build-release\mLLM.exe" (
    echo.
    echo 오류: mLLM.exe 가 없습니다. 먼저 빌드하세요.
    echo.
    echo   1. setup.bat 실행 ^(환경 설정^)
    echo   2. build_mllm.bat 실행 ^(빌드^)
    echo.
    pause
    exit /b 1
)

echo.
echo 모델: %MODEL%
echo 종료하려면 'q' 입력 후 엔터
echo.

set PATH=%~dp0cmake-build-release;%PATH%
"%~dp0cmake-build-release\mLLM.exe" %MODEL% %2 %3 %4
