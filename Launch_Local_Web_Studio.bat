@echo off
title EGFR-C797S Local Web Studio - Punjabi University, Patiala
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    start "" http://127.0.0.1:8000
    "%~dp0.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
    exit /b 0
)

if exist "C:\Users\owner\.gemini\antigravity\scratch\egfr_c797s_qsar_llm\.venv\Scripts\python.exe" (
    start "" http://127.0.0.1:8000
    "C:\Users\owner\.gemini\antigravity\scratch\egfr_c797s_qsar_llm\.venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
    exit /b 0
)

echo Please install dependencies first via Launch_EGFR_C797S_Standalone_Tool.bat or pip install -r requirements.txt fastapi uvicorn jinja2 python-multipart
pause
