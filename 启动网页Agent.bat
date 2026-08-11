@echo off
cd /d "%~dp0"
title Financial Research Agent

if not exist ".venv\Scripts\activate.bat" (
  echo [错误] 未找到 .venv，请先创建虚拟环境并安装依赖。
  pause
  exit /b 1
)

call .venv\Scripts\activate
echo 正在启动网页 Agent...
echo 浏览器打开: http://127.0.0.1:8000/agent
echo 关闭本窗口即停止服务。
echo.

start "" "http://127.0.0.1:8000/agent"
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
pause
