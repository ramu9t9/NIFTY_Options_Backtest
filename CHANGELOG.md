# Changelog

All notable changes to the NIFTY Options Paper Trading System will be documented in this file.

## [2.0.0] - 2026-01-09

### Added - Backtest Engine
- **New `backtest_engine.py`**: Complete backtest engine implementation
  - Reuses proven strategy logic from `live_trading_engine.py`
  - 3-stage strategy: Trend Detection → Pattern Analysis → Trade Execution
  - Achieved 97.4% match with original backtest (111/114 trades)
  - Generated 150 trades with 52% win rate and ₹858K profit
  
- **Backtest Improvements**:
  - Data availability pre-checking (eliminates invalid signals)
  - Independent pattern window (zero overlap with signal candle)
  - Incomplete candle filtering (minimum 3 ticks per candle)
  - Industry-standard clock-aligned candles using `dt.floor()`

- **Test Script**: `test_backtest_engine.py` for standalone validation

### Added - Documentation
- **`Documents/Trading_Strategy_Complete_Guide.md`**: Comprehensive strategy documentation
  - Complete 3-stage strategy explanation
  - Real-world examples with actual log outputs
  - All threshold values and parameters
  - Risk management rules
  
- **`Documents/backtest_engine_corrections.md`**: Technical review and responses
- **`Documents/claude_backtest_handoff.md`**: Detailed handoff documentation

### Fixed - Dashboard
- **Log Filtering**: Fixed Pattern Analysis and Signal Generation tab filters
  - Pattern Analysis now shows candle building and trend detection
  - Signal Generation shows only signal events and pattern detection
  - Swapped filter keywords for distinct content

### Changed
- Updated `README.md` with backtest engine information
- Improved code comments and documentation throughout
- Cleaned up temporary analysis scripts

### Technical Details
- **Database**: `G:\Projects\Centralize Data Centre\data\nifty_local.db`
- **Date Range Tested**: 2025-08-29 to 2026-01-01
- **Ticks Processed**: 355,761 NIFTY 50 ticks
- **Candles Built**: 59,507 30-second candles
- **Trends Detected**: 150 (with data availability check)

---

## [1.0.0] - 2025-12-XX

### Initial Release
- Live trading engine with 3-stage strategy
- NiceGUI dashboard with real-time monitoring
- Paper trading mode with trade tracking
- Database integration for tick data
- Broadcaster integration for real-time data

### Features
- **Live Trading Engine**: Real-time trend detection and pattern analysis
- **Dashboard**: Multi-tab interface with live charts and logs
- **Paper Trading**: Simulated trading with P&L tracking
- **Data Management**: SQLite database for historical data
- **Broadcaster**: Real-time data collection from Angel One API

---

## Version History

- **v2.0.0** (2026-01-09): Backtest engine + improvements
- **v1.0.0** (2025-12-XX): Initial live trading system
