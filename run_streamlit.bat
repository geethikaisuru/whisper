@echo off
REM Streamlit Live Transcription Launcher for Windows
REM This batch file will launch the Streamlit application

echo 🎤 Streamlit Live Transcription Launcher
echo =========================================

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Try to run the launcher script first
echo 🚀 Starting Streamlit Live Transcription...
python run_streamlit.py

REM If that fails, try direct Streamlit launch
if %ERRORLEVEL% neq 0 (
    echo.
    echo ⚠️  Launcher failed, trying direct Streamlit launch...
    streamlit run streamlit_transcription.py
)

echo.
echo 👋 Application closed. Press any key to exit...
pause >nul 