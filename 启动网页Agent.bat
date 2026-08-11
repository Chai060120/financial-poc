@echo off
cd /d "%~dp0"
title Financial Web Agent

if not exist ".venv\Scripts\python.exe" (
  echo [错误] 未找到 .venv，请先创建虚拟环境并安装依赖。
  pause
  exit /b 1
)

echo 正在启动网页 Agent...
echo 浏览器将打开 http://127.0.0.1:8000/agent
echo 关闭本窗口即停止服务。
echo.

start "" "http://127.0.0.1:8000/agent"
".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
