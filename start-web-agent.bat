@echo off
cd /d "%~dp0"
title Financial Research Agent

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found.
  echo Please run:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

echo Starting web Agent...
echo Open: http://127.0.0.1:8000/agent
echo Close this window to stop the server.
echo.

start "" "http://127.0.0.1:8000/agent"
".venv\Scripts\python.exe" -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
if errorlevel 1 (
  echo.
  echo [ERROR] Failed to start. See messages above.
)
pause
