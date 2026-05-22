@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 > nul 2>&1
cd /d C:\workspace\git\mLLM\cmake-build-release
"C:\Program Files\JetBrains\CLion 2026.1\bin\ninja\win\x64\ninja.exe" mLLM > C:\workspace\git\mLLM\cmake-build-release\build_ninja.log 2>&1
echo %ERRORLEVEL% > C:\workspace\git\mLLM\cmake-build-release\build_exit.txt
