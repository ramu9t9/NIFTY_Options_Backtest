# NIFTY Options Paper Trading System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

A comprehensive **paper trading system** for NIFTY options with both **live trading** and **backtesting** capabilities.

### Key Features

- **Live Paper Trading**: Real-time trend detection and pattern analysis
- **Backtest Engine**: Historical strategy validation on tick data
- **NiceGUI Dashboard**: Real-time monitoring and trade management
- **Proven Strategy**: 3-stage approach with documented results

### Strategy Performance

**Backtest Results** (Aug 29, 2025 - Jan 1, 2026):
- **150 trades** | **52% win rate** | **₹858K profit** | **1.80 profit factor**
- **97.4% match** with original backtest (111/114 trades)
- Average hold time: **1.5 minutes**

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements_dashboard.txt
```

### Backtest Mode (Recommended First Step)

```bash
# Run backtest on historical data
python test_backtest_engine.py
```

This validates the strategy on historical data before live trading.

### Live Trading Mode

```batch
# Terminal 1: Start data collector
scripts\start_live_data_collector.bat

# Terminal 2: Start paper trading dashboard
scripts\start_live_paper_trading.bat
```

Open http://localhost:8080 in your browser.

---

## 📁 Project Structure

```
NIFTY_Options_Backtest/
├── backtest_engine.py          # Backtest engine (NEW)
├── test_backtest_engine.py     # Backtest test script
├── live_trading_engine.py      # Live trading strategy
├── live_dashboard.py           # NiceGUI dashboard
├── trade_store.py              # Trade data management
│
├── Documents/
│   ├── Trading_Strategy_Complete_Guide.md  # Strategy documentation
│   ├── claude_backtest_handoff.md          # Backtest specifications
│   └── backtest_engine_corrections.md      # Technical review
│
├── scripts/                    # Startup scripts
├── broadcaster/                # Data collection
├── paper_trading/             # Trade tracking
└── vps_data_collector/        # VPS data sync
```

---

## 🎯 Strategy Details

### 3-Stage Approach

#### Stage 1: Trend Detection
- **30-second candles** from NIFTY 50 tick data
- **0.11% cumulative move** threshold
- Direction tracking (UP/DOWN/NEUTRAL)

#### Stage 2: Pattern Analysis
- **60-second window** of options Greeks
- **Fast indicators**: IV, Delta, Volume Ratio, Premium Momentum
- **Thresholds**:
  - IV Change: ±5%
  - Volume Ratio Change: ±10%
  - Delta Change: ±0.03
  - Premium Momentum: ±2%

#### Stage 3: Trade Execution
- **Entry**: ATM options (CE/PE) based on pattern direction
- **Exits**:
  - Target: +10%
  - Stop Loss: -5%
  - Time: 3 minutes max hold
- **Lot Size**: 3750 (50 lots × 75 qty)

---

## 🔬 Backtest Engine

### Features

✅ **Reuses Live Strategy Logic**: Same code as `live_trading_engine.py`  
✅ **Data Availability Pre-Check**: Eliminates invalid signals  
✅ **Independent Pattern Window**: Zero overlap with signal candle  
✅ **Industry-Standard Candles**: Clock-aligned 30-second intervals  
✅ **Complete Trade Simulation**: Entry, exits, P&L with transaction costs

### Usage

```python
from backtest_engine import BacktestEngine

# Initialize
engine = BacktestEngine(
    db_path=r"G:\Projects\Centralize Data Centre\data\nifty_local.db",
    config={
        'candle_interval_seconds': 30,
        'movement_threshold': 0.11,
        'pattern_window_seconds': 60,
        'lot_size': 3750,
        'target_pct': 10.0,
        'stop_pct': 5.0,
        'max_hold_minutes': 3.0
    }
)

# Run backtest
results = engine.run(start_date="2025-08-29", end_date="2026-01-01")

# View results
print(f"Total Trades: {results.total_trades}")
print(f"Win Rate: {results.win_rate:.2f}%")
print(f"Net P&L: ₹{results.total_net_pnl:,.2f}")
print(f"Profit Factor: {results.profit_factor:.2f}")
```

### Validation

The backtest engine was validated against proven results:

| Metric | Expected | Achieved | Match |
|--------|----------|----------|-------|
| Trade Timestamps | 114 | 111 | **97.4%** |
| Total Trades | 114 | 150 | +36 extra |
| Win Rate | 57.89% | 52.00% | -5.89% |
| Profit | ₹1.04M | ₹858K | -17.4% |

**Extra 39 Trades**: Profitable (₹197K, 56.4% WR) - valid based on data availability.

---

## 📊 Live Dashboard

### Features

- **Real-Time Monitoring**: Live feed status, connection tracking
- **Multi-Tab Interface**:
  - Overview: System status and quick stats
  - Candle Building: 30-second OHLC construction
  - Pattern Analysis: Trend detection and cumulative moves
  - Signal Generation: Pattern confirmation and trade signals
  - Trade Execution: Active trades and P&L tracking
  - Trade History: Complete trade log with filters
  
- **Live Charts**: Real-time NIFTY price and trade P&L
- **Log Streaming**: Filtered logs for each stage

### Access

```
http://localhost:8080
```

---

## 🗄️ Database

### Structure

**Database**: `nifty_local.db` (SQLite)

**Table**: `ltp_ticks`
- `ts`: Timestamp (ISO format with timezone)
- `symbol`: NIFTY 50 or option symbol
- `ltp`: Last traded price
- `volume`, `oi`: Volume and open interest
- `iv`, `delta`, `gamma`, `theta`, `vega`: Greeks

### Data Source

- **Live**: Angel One API via broadcaster
- **Historical**: VPS data collector sync

---

## 📚 Documentation

### Strategy Documentation
- **[Trading_Strategy_Complete_Guide.md](Documents/Trading_Strategy_Complete_Guide.md)**: Complete strategy explanation with examples

### Backtest Documentation
- **[claude_backtest_handoff.md](Documents/claude_backtest_handoff.md)**: Backtest specifications and requirements
- **[backtest_engine_corrections.md](Documents/backtest_engine_corrections.md)**: Technical review and responses

### Handoff Documents
- **[chatgpt_handoff.md](chatgpt_handoff.md)**: Live trading system handoff

---

## 🔧 Configuration

### Strategy Parameters

Edit in `backtest_engine.py` or `live_trading_engine.py`:

```python
config = {
    'candle_interval_seconds': 30,      # Candle size
    'movement_threshold': 0.11,         # Trend threshold (%)
    'pattern_window_seconds': 60,       # Pattern analysis window
    'lot_size': 3750,                   # Position size
    'target_pct': 10.0,                 # Target profit (%)
    'stop_pct': 5.0,                    # Stop loss (%)
    'max_hold_minutes': 3.0,            # Max hold time
}
```

### Pattern Thresholds

Edit in `live_trading_engine.py`:

```python
THRESHOLDS_DEFAULT = {
    "iv_change_pct": 5.0,              # IV change threshold
    "volume_ratio_change": 10.0,       # Volume ratio change
    "delta_change": 0.03,              # Delta change
    "premium_momentum": 2.0,           # Premium momentum (%)
}
```

---

## 🧪 Testing

### Run Backtest

```bash
python test_backtest_engine.py
```

### Test Live Engine (Replay Mode)

```bash
# Terminal 1: Start broadcaster writer
scripts\start_broadcaster_writer.bat

# Terminal 2: Start dashboard
scripts\start_live_dashboard.bat
```

---

## 📈 Results

### Backtest Performance (Aug 29, 2025 - Jan 1, 2026)

- **Data Processed**: 355,761 NIFTY ticks
- **Candles Built**: 59,507 (30-second)
- **Trends Detected**: 150 (with data availability check)
- **Trades Executed**: 150
- **Win Rate**: 52.00%
- **Total Net P&L**: ₹858,359
- **Profit Factor**: 1.80
- **Avg Win**: ₹24,801
- **Avg Loss**: ₹14,946
- **Max Win**: ₹61,132
- **Max Loss**: ₹40,140
- **Avg Hold Time**: 1.5 minutes

### Exit Breakdown

- **TARGET**: 40% of trades
- **STOP_LOSS**: 35% of trades
- **TIME**: 25% of trades

---

## 🛠️ Development

### Recent Improvements (v2.0.0)

1. ✅ **Data Availability Pre-Check**: Eliminates 23 invalid signals
2. ✅ **Independent Pattern Window**: Zero overlap with signal candle
3. ✅ **Incomplete Candle Filtering**: Minimum 3 ticks per candle
4. ✅ **Comprehensive Documentation**: Complete strategy guide

### Code Quality

- **Type Hints**: Full type annotations
- **Logging**: Detailed logging at all stages
- **Error Handling**: Robust error handling
- **Comments**: Inline documentation

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is a personal trading system. For questions or suggestions, please open an issue.

---

## ⚠️ Disclaimer

This is a **paper trading system** for educational and testing purposes only. 

**NOT FOR LIVE TRADING** without proper risk management and testing.

Past performance does not guarantee future results.

---

## 📞 Support

For issues or questions:
1. Check documentation in `Documents/`
2. Review `CHANGELOG.md` for recent changes
3. Open an issue on GitHub

---

**Version**: 2.0.0  
**Last Updated**: January 9, 2026
