@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ========================================
echo   WhisperX + pyannote Subtitle Generator
echo ========================================
echo.

set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ffmpeg not found. Please install ffmpeg.
    pause
    exit /b 1
)

set "VIDEO_FILE="
for %%f in (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v) do (
    set "VIDEO_FILE=%%~f"
    goto :found
)
:found

if "%VIDEO_FILE%"=="" (
    echo [ERROR] No video file found in current directory.
    echo Please place a video file here, or run manually:
    echo   venv\Scripts\python.exe subtitle_generator.py ^<video^>
    pause
    exit /b 1
)

echo Found: %VIDEO_FILE%
echo.

set "SPEAKER_FLAG="
if exist ".hf_token" (
    echo [.hf_token found - speaker diarization enabled]
    set "SPEAKER_FLAG=--speakers"
) else (
    echo [No .hf_token - speaker diarization disabled]
)

echo.
echo Processing...
echo.

"%PYTHON%" -X utf8 subtitle_generator.py "%VIDEO_FILE%" --device cpu %SPEAKER_FLAG%

echo.
echo ========================================
pause
