@echo off
title EGFR-C797S Standalone Tool - Punjabi University, Patiala
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0desktop_app.py"
    exit /b 0
)

if exist "C:\Users\owner\.gemini\antigravity\scratch\egfr_c797s_qsar_llm\.venv\Scripts\pythonw.exe" (
    start "" "C:\Users\owner\.gemini\antigravity\scratch\egfr_c797s_qsar_llm\.venv\Scripts\pythonw.exe" "%~dp0desktop_app.py"
    exit /b 0
)

echo First-time setup: Creating local Python virtual environment for EGFR-C797S Standalone Tool...
python -m venv "%~dp0.venv"
"%~dp0.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0desktop_app.py"
