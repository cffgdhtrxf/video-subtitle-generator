@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ========================================
echo   WhisperX + pyannote Setup Wizard
echo ========================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)
echo Done.
echo.

echo [2/3] Installing dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install whisperx pyannote.audio torch torchaudio
if errorlevel 1 (
    echo [ERROR] pip install failed
    pause
    exit /b 1
)
echo Done.
echo.

echo [3/3] HuggingFace Token (for speaker diarization)
echo.
echo If you need speaker diarization:
echo   1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1 - Accept terms
echo   2. Visit https://huggingface.co/pyannote/segmentation-3.0 - Accept terms
echo   3. Visit https://huggingface.co/settings/tokens - Create token
echo.
set /p HF_TOKEN="Enter HF Token (or press Enter to skip): "

if not "%HF_TOKEN%"=="" (
    echo %HF_TOKEN%> .hf_token
    echo Token saved to .hf_token
) else (
    echo Skipped.
)
echo.

echo ========================================
echo   Setup complete!
echo   Place a video in this folder and run: run.bat
echo ========================================
pause
