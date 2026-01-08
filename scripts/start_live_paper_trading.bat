@echo off
REM ========================================
REM   NIFTY Paper Trading - LIVE MODE
REM ========================================
echo.
echo ========================================
echo   NIFTY Paper Trading - LIVE MODE
echo ========================================
echo.
echo Database: nifty_live.db
echo Mode: Live (with Angel One WebSocket for exits)
echo.
echo Starting paper trading engine...
echo Logs will be saved to: logs\live_trading_engine.log
echo.

REM Change to project root
cd /d "%~dp0.."

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Run with output redirected to log file
py paper_trading/paper_trading_engine.py --db-path "g:\Projects\NIFTY_Options_Backtest\data\nifty_live.db" --log-level INFO > logs\live_trading_engine.log 2>&1

echo.
echo Trading engine stopped. Check logs\live_trading_engine.log for details.
pause
