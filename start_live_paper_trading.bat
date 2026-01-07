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
echo.

python paper_trading/paper_trading_engine.py --db-path "g:\Projects\NIFTY_Options_Backtest\data\nifty_live.db" --log-level INFO

pause
