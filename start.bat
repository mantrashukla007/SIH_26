@echo off
title BIS Intelligent Assistant
color 0A

echo.
echo  ============================================================
echo   BIS Intelligent Assistant — Startup
echo  ============================================================
echo.

:: Kill anything already on ports 8000 and 3000
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8000 "') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":3000 "') do (
    taskkill /PID %%p /F >nul 2>&1
)

echo  [1/2] Starting Backend API on http://127.0.0.1:8000 ...
start "BIS Backend" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

:: Wait 3 seconds for the backend to initialize
timeout /t 3 /nobreak >nul

echo  [2/2] Starting Frontend on http://localhost:3000 ...
start "BIS Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

:: Wait another second then open the browser
timeout /t 2 /nobreak >nul

echo.
echo  ============================================================
echo   Both servers running! Opening browser...
echo  ============================================================
echo.
echo   App    : http://localhost:3000
echo   API    : http://127.0.0.1:8000
echo   Docs   : http://127.0.0.1:8000/docs
echo  ============================================================
echo.

start "" "http://localhost:3000"

pause
