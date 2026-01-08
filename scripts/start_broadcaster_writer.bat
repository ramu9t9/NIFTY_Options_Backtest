@echo off
REM ========================================
REM   Broadcaster Data Writer - REPLAY MODE
REM ========================================
echo.
echo ========================================
echo   Broadcaster Data Writer - REPLAY MODE
echo ========================================
echo.
echo Broadcaster: ws://localhost:8765
echo Database: nifty_replay.db
echo.
echo Connecting to broadcaster...
echo.

REM Change to project root
cd /d "%~dp0.."

py broadcaster_data_writer.py --broadcaster-url ws://localhost:8765 --db-path "g:\Projects\NIFTY_Options_Backtest\data\nifty_replay.db"

pause
