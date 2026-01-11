# Index Options Backtest Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

A **modular, production-quality Index Options Backtest Engine** for testing trading strategies on historical NIFTY options data. The engine is designed with clean architecture to allow adding multiple strategies without rewriting core logic.

### Key Features

- **Modular Architecture**: Pluggable strategy system with registry
- **SQLite Integration**: Direct support for tick-level historical data
- **Full Options Chain Simulation**: Realistic contract selection, execution, and mark-to-market
- **Risk Management**: Built-in position sizing, risk limits, and exposure controls
- **Production Quality**: Type hints, docstrings, comprehensive unit tests
- **NiceGUI Dashboard**: Interactive web UI for running backtests and comparing results
- **CLI Interface**: Command-line runner with config file support

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
pip install -r requirements.txt

# Or for dashboard only (lighter install)
pip install -r requirements_dashboard.txt
```

### Database Setup

The engine expects a SQLite database at:
```
G:\Projects\Centralize Data Centre\data\nifty_local.db
```

The database should contain a `ltp_ticks` table with 5-second tick data for:
- NIFTY 50 index (symbol: `NIFTY 50`)
- Options contracts (symbols like `NIFTY25116CE25000`)
- Futures contracts (symbols like `NIFTY25JANFUT`)

### Run a Backtest

#### Via NiceGUI Dashboard (Recommended)

```bash
py apps/nicegui_app.py
```

Then open `http://localhost:8080` in your browser. The dashboard provides:
- Strategy selection and parameter editing
- Config file management
- Real-time backtest progress
- Results visualization (equity curves, metrics, trade logs)
- Run comparison tools

#### Via CLI

```bash
# Using a config file
py -m index_options_bt.run.cli --config configs/breakout_nifty_sqlite.yaml --start 2025-01-10 --end 2025-01-11

# With CLI overrides
py -m index_options_bt.run.cli --config configs/ema_crossover_nifty_sqlite.yaml --set strategy.params.fast_period=15
```

#### Programmatically

```python
from index_options_bt.config import load_config
from index_options_bt.run.runner import run_backtest

# Load config
cfg = load_config("configs/breakout_nifty_sqlite.yaml")

# Override dates
cfg.engine.start = "2025-01-10"
cfg.engine.end = "2025-01-11"

# Run backtest
results = run_backtest(cfg)
print(f"Total Trades: {results['total_trades']}")
print(f"Net P&L: ₹{results['net_pnl']:,.2f}")
```

---

## 📁 Project Structure

```
NIFTY_Options_Backtest/
├── index_options_bt/          # Main backtest engine package
│   ├── config/                # Configuration schemas and loaders
│   ├── data/                  # Data providers (SQLite, bars, calendars)
│   ├── strategy/              # Strategy base classes and implementations
│   ├── execution/             # Contract selection and execution simulation
│   ├── portfolio/             # Portfolio management and PnL tracking
│   ├── risk/                  # Risk management and position sizing
│   └── run/                   # Backtest runner, CLI, and artifacts
│
├── apps/
│   └── nicegui_app.py        # Interactive web dashboard
│
├── configs/                   # Strategy configuration files (YAML)
│   ├── breakout_nifty_sqlite.yaml
│   ├── ema_crossover_nifty_sqlite.yaml
│   └── sample_strangle_sqlite.yaml
│
├── examples/                  # Example scripts
│   └── run_backtest.py
│
├── tests/                     # Unit tests
│   ├── test_config_loader.py
│   ├── test_contract_selector.py
│   ├── test_execution_model.py
│   ├── test_portfolio_pnl.py
│   ├── test_registry.py
│   ├── test_risk_manager.py
│   ├── test_run_artifacts.py
│   ├── test_sqlite_provider.py
│   └── test_backend.py        # Backend testing script
│
├── runs/                      # Backtest run outputs
│   └── run-YYYYMMDD-HHMMSS-XXX/
│       ├── config_resolved.json
│       ├── equity_curve.csv
│       ├── trades.csv
│       ├── positions.csv
│       ├── selection.csv
│       ├── metrics.json
│       └── run.log
│
├── reports/                   # Excel exports (IST datetime)
│   └── run-YYYYMMDD-HHMMSS-XXX.xlsx
│
├── data/                      # Local database files (optional)
│   ├── nifty_live.db
│   └── nifty_replay.db
│
├── Archive/                   # Archived old files (for reference)
│
├── export_excel.py            # Excel export CLI script
├── README.md                  # This file
└── requirements_dashboard.txt # Python dependencies
```

---

## 🎯 Architecture

### Core Concepts

1. **DataProvider**: Fetches market data (spot, futures, options chain) for a given timestamp
2. **Strategy**: Generates trading "intents" (LONG_CALL, LONG_PUT, FLAT) based on spot-derived features
3. **Contract Selector**: Converts intents into specific option contracts based on criteria (ATM, DTE, Delta, liquidity)
4. **Execution Model**: Simulates trade fills with slippage, commissions, and fees
5. **Portfolio**: Tracks positions, calculates PnL (realized/unrealized), handles expiry
6. **Risk Manager**: Enforces position limits, notional caps, and loss limits

### Event Loop

For each timestamp (bar) in the backtest:
1. **Pull Data**: Fetch spot, futures, and options chain snapshot
2. **Compute Features**: Calculate indicators (e.g., EMAs, breakouts, volume ratios)
3. **Generate Intents**: Strategy produces trading intents based on features
4. **Risk Management**: Check position limits and size positions
5. **Contract Selection**: Select specific option contracts for intents
6. **Execution**: Simulate order fills
7. **Portfolio Update**: Apply fills, calculate MTM, check TP/SL
8. **Expiry Handling**: Close/settle expiring contracts

---

## 📝 Available Strategies

### 1. Breakout Strategy (`example_breakout`)

Detects price breakouts above/below recent highs/lows.

**Parameters:**
- `lookback_bars`: Number of bars to look back (default: 20)
- `breakout_threshold_pct`: Percentage move to trigger breakout (default: 0.05%)
- `take_profit_pct`: Take profit percentage (default: 10.0%)
- `stop_loss_pct`: Stop loss percentage (default: 5.0%)

**Config:** `configs/breakout_nifty_sqlite.yaml`

### 2. EMA Crossover Strategy (`ema_crossover`)

Generates signals based on fast/slow EMA crossovers.

**Parameters:**
- `fast_period`: Fast EMA period (default: 12)
- `slow_period`: Slow EMA period (default: 26)
- `take_profit_pct`: Take profit percentage (default: 10.0%)
- `stop_loss_pct`: Stop loss percentage (default: 5.0%)

**Config:** `configs/ema_crossover_nifty_sqlite.yaml`

### 3. Strangle Strategy (`strangle`)

Multi-leg options selling strategy (example for future extension).

**Config:** `configs/sample_strangle_sqlite.yaml`

---

## 🔧 Configuration

Configuration files use YAML format and support:
- Strategy selection and parameters
- Data source settings
- Contract selection criteria
- Risk management limits
- Execution parameters
- Date ranges

### Example Config

```yaml
engine:
  provider: "sqlite"
  db_path: "G:/Projects/Centralize Data Centre/data/nifty_local.db"
  start: "2025-01-10"
  end: "2025-01-11"
  bar_size_seconds: 30

strategy:
  name: "ema_crossover"
  params:
    fast_period: 12
    slow_period: 26
    take_profit_pct: 10.0
    stop_loss_pct: 5.0

selector:
  mode: "atm"
  expiry_preference: "nearest"
  contract_multiplier: 75
  liquidity:
    min_bid: 0.0
    max_spread_pct: 1.0

risk:
  max_notional: 1000000.0
  max_loss_per_trade: 50000.0
  max_loss_per_day: 100000.0

execution:
  slippage_bps: 5.0
  commission_per_contract: 20.0
  fallback_price: "ltp"
```

### Environment Overrides

Override config values via environment variables:
```bash
export ENGINE_START="2025-01-10"
export STRATEGY_PARAMS__FAST_PERIOD=15
export RISK__MAX_NOTIONAL=2000000.0
```

### CLI Overrides

Override config values via CLI flags:
```bash
py -m index_options_bt.run.cli --config configs/ema_crossover_nifty_sqlite.yaml \
  --set strategy.params.fast_period=15 \
  --set risk.max_notional=2000000.0
```

---

## 🧪 Testing

Run all tests:
```bash
py -m pytest tests/ -v
```

Run specific test file:
```bash
py -m pytest tests/test_contract_selector.py -v
```

Run with coverage:
```bash
py -m pytest tests/ --cov=index_options_bt --cov-report=html
```

---

## 📊 Run Artifacts

Each backtest run generates standardized outputs in `runs/run-<timestamp>/`:

- **`config_resolved.json`**: Final resolved configuration (with overrides applied)
- **`equity_curve.csv`**: Portfolio value over time
- **`trades.csv`**: Complete trade log (entry/exit times, prices, PnL)
- **`positions.csv`**: Position history (snapshots at each bar)
- **`selection.csv`**: Contract selection log (which contracts were chosen and why)
- **`metrics.json`**: Summary statistics (win rate, profit factor, max drawdown, etc.)
- **`run.log`**: Detailed execution log
- **`report.png`**: Equity curve visualization

---

## 🔌 Adding New Strategies

1. **Create Strategy Class**:

```python
from index_options_bt.strategy.base import Strategy, Intent
from index_options_bt.strategy.registry import register_strategy

@register_strategy("my_strategy")
class MyStrategy(Strategy):
    def __init__(self, params: dict):
        super().__init__(params)
        self.param1 = params.get("param1", 10)
    
    def generate_intents(self, snapshot: MarketSnapshot, features: dict) -> list[Intent]:
        # Your strategy logic here
        if some_condition:
            return [Intent(
                direction="LONG_CALL",
                quantity=1,
                metadata={"take_profit_pct": 10.0, "stop_loss_pct": 5.0}
            )]
        return []
```

2. **Create Config File**:

```yaml
strategy:
  name: "my_strategy"
  params:
    param1: 10
    take_profit_pct: 10.0
    stop_loss_pct: 5.0
```

3. **Run Backtest**:

```bash
py apps/nicegui_app.py
# Select your config and run!
```

---

## 📚 Documentation

- **Strategy Guide**: See strategy docstrings and example configs
- **API Reference**: Inline docstrings for all classes and methods
- **Archived Docs**: See `Archive/old_docs/` for historical documentation

---

## 🛠️ Development

### Code Quality

- **Type Hints**: Full type annotations throughout
- **Docstrings**: Google-style docstrings for all public APIs
- **Testing**: Comprehensive unit tests with pytest
- **Linting**: Use `ruff` or `black` for formatting

### Contributing

This is a personal trading system. For questions or suggestions, please open an issue.

---

## ⚠️ Disclaimer

This is a **backtesting framework** for educational and research purposes only.

**NOT FOR LIVE TRADING** without proper risk management, testing, and compliance review.

Past performance does not guarantee future results. All trading involves risk.

---

## 📞 Support

For issues or questions:
1. Check the inline docstrings and example configs
2. Review the test files for usage examples
3. Check `Archive/old_docs/` for historical context

---

**Version**: 3.0.0 (Modular Engine)  
**Last Updated**: January 11, 2026

