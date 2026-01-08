@echo off
REM ========================================
REM   NIFTY Paper Trading - REPLAY MODE
REM ========================================
echo.
echo ========================================
echo   NIFTY Paper Trading - REPLAY MODE
echo ========================================
echo.
echo Database: nifty_replay.db
echo Mode: Replay (NO WebSocket needed)
echo.
echo Starting paper trading engine...
echo.

REM Change to project root
cd /d "%~dp0.."

py paper_trading/paper_trading_engine.py --db-path "g:\Projects\NIFTY_Options_Backtest\data\nifty_replay.db" --log-level INFO

pause
