@echo off
echo ===============================================
echo  Faster-Whisper Installation Helper
echo ===============================================
echo.

echo Installing faster-whisper...
pip install faster-whisper

echo.
echo Testing installation...
python install_faster_whisper.py

echo.
echo Installation complete!
echo You can now run: streamlit run streamlit_transcription.py
echo.
pause 