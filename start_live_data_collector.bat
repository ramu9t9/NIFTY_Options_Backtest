@echo off
REM ========================================
REM   NIFTY Data Collector - LIVE MODE
REM ========================================
echo.
echo ========================================
echo   NIFTY Data Collector - LIVE MODE
echo ========================================
echo.
echo Database: nifty_live.db
echo Mode: Real-time Angel One API
echo.
echo Starting data collector...
echo.

cd vps_data_collector
python nifty_stream_local_sqlite.py

pause
