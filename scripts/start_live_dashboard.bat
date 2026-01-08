@echo off
setlocal enabledelayedexpansion

REM Double-click launcher for NiceGUI Live Dashboard
REM Opens on: http://localhost:8080
REM Logs to: logs\live_dashboard_<timestamp>.log

REM Get the parent directory (project root)
cd /d "%~dp0.."

if not exist "logs" mkdir "logs"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i
set LOG_FILE=logs\live_dashboard_%TS%.log

echo ================================================================================>> "%LOG_FILE%"
echo START LIVE DASHBOARD %DATE% %TIME%>> "%LOG_FILE%"
echo URL: http://localhost:8080>> "%LOG_FILE%"
echo ================================================================================>> "%LOG_FILE%"

echo Starting NiceGUI Dashboard...
echo - Open: http://localhost:8080
echo - Log file: %LOG_FILE%
echo.
echo If it doesn't open automatically, copy-paste the URL into your browser.
echo To stop: close this window or press Ctrl+C.
echo.

py -m pip show nicegui >nul 2>&1
if errorlevel 1 (
  echo NiceGUI not installed. Installing from requirements_dashboard.txt ...
  py -m pip install -r requirements_dashboard.txt
)

py live_dashboard.py >> "%LOG_FILE%" 2>&1

echo.
echo Dashboard stopped.
echo Logs saved to: %LOG_FILE%
pause


