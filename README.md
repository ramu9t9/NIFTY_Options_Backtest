# NIFTY Options Live Paper Trading System

## 📊 Project Overview

This is a **live paper trading system** for NIFTY options strategy that uses:
- **Real-time trend detection** (30-second candles, 0.11% cumulative move threshold)
- **Pattern analysis** (60-second window, fast indicators: IV, Delta, Volume, Premium)
- **Hybrid architecture**: DB polling for signals + Angel One WebSocket for active trade tracking
- **NiceGUI Dashboard** for real-time monitoring and trade management

**Strategy Parameters**:
- Entry: ATM options (CE/PE) based on pattern direction
- Exit: +10% target, -5% stop loss, or 3 minutes max hold
- Lot size: 3750 (50 lots × 75 quantity) - configurable

---

## 🚀 Quick Start

### Live Mode (Real-time Trading)
```batch
# Terminal 1: Start data collector
scripts\start_live_data_collector.bat

# Terminal 2: Start paper trading
scripts\start_live_paper_trading.bat
```

### Replay Mode (Historical Testing)
```batch
# Terminal 1: Start broadcaster data writer
scripts\start_broadcaster_writer.bat

# Terminal 2: Start dashboard
scripts\start_live_dashboard.bat
# Then open http://localhost:8080
```

> 📖 **Detailed instructions**: See `scripts/README.md`

---

## 📁 Project Structure

### Core Components

#### `live_dashboard.py`
NiceGUI web dashboard for live paper trading:
- Real-time feed monitoring (IST timestamps from broadcaster)
- Connection status and reconnect tracking
- Active trade panel (entry, TG/SL, running P&L, hold time)
- Trade history with SQLite database integration
- Export to Excel/CSV (safe with Excel open)
- Single active trade enforcement

**Quick Start**: Double-click `start_live_dashboard.bat` or run:
```powershell
py live_dashboard.py
```
Dashboard opens at `http://localhost:8080` (default)

#### `paper_trading/` - Paper Trading Engine
Standalone paper trading system (DB-based, no broadcaster required):

- **`paper_trading_engine.py`**: Main orchestrator
  - Polls SQLite DB for trend signals (5-second cadence)
  - Pattern analysis on 60-second window
  - Angel One WebSocket for real-time exit monitoring
  - Writes trades to timestamped CSV

- **`realtime_trend_detector.py`**: Detects 0.11% cumulative moves on 30s candles
- **`realtime_pattern_analyzer.py`**: Analyzes fast indicators and generates trade signals
- **`websocket_handler.py`**: Manages Angel One SmartWebSocketV2 connection
- **`performance_tracker.py`**: Calculates rolling performance statistics

**Quick Start**: Double-click `start_live_paper_trading.bat` or run:
```powershell
py paper_trading\paper_trading_engine.py --db-path "G:\Projects\Centralize Data Centre\data\nifty_local.db"
```

#### `live_trading_engine.py`
Reusable core engine with:
- Candle building (30-second intervals)
- Trend signal detection
- Pattern calculation functions
- Transaction cost calculation (Angel One formula)

#### `trade_store.py`
SQLite database operations for storing and retrieving trade records.

#### `broadcaster_client.py`
WebSocket client for connecting to the Centralize Data Centre broadcaster.

### Supporting Components

#### `vps_data_collector/`
VPS data collector scripts:
- `nifty_stream_local_sqlite.py`: Collects NIFTY data and writes to SQLite
- `market_scheduler.py`: Market hours scheduler

#### `broadcaster/`
Documentation for broadcaster integration:
- `Broadcast Documents/`: How to use broadcaster
- `Broadcaster_Integration/`: WebSocket client integration guide

#### `exports/`
SQLite database for trade history (`live_trades.db`)

---

## 🚀 Quick Start

### Option 1: NiceGUI Dashboard (Recommended)

1. **Start Dashboard**:
   ```powershell
   # Double-click or run:
   start_live_dashboard.bat
   ```

2. **Configure** (if needed):
   - Open `http://localhost:8080`
   - Adjust strategy parameters (defaults are pre-set)
   - Click "Reset to Strategy Defaults" to restore defaults

3. **Start Trading**:
   - Click "Connect & Start Paper Trading"
   - Monitor active trades and history in dashboard

### Option 2: Standalone Paper Trading Engine

1. **Start Engine**:
   ```powershell
   py paper_trading\paper_trading_engine.py --db-path "G:\Projects\Centralize Data Centre\data\nifty_local.db" --log-level INFO
   ```

2. **Monitor Output**:
   - Trades written to `paper_trading/paper_trades_YYYYMMDD_HHMMSS.csv`
   - Console logs show signals, entries, exits

---

## ⚙️ Configuration

### Strategy Defaults (in Dashboard)

- **Candle Interval**: 30 seconds
- **Movement Threshold**: 0.11% (cumulative move)
- **Pattern Window**: 60 seconds
- **Target**: +10%
- **Stop Loss**: -5%
- **Max Hold**: 3 minutes
- **Lot Size**: 3750 (50 lots)

### Environment Variables

Create `.env` file or set system environment variables for Angel One API:

```env
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret
```

Or use non-prefixed versions:
```env
API_KEY=your_api_key
CLIENT_ID=your_client_id
PASSWORD=your_password
TOTP_SECRET=your_totp_secret
```

### Database Path

Default DB path: `G:\Projects\Centralize Data Centre\data\nifty_local.db`

Update in:
- Dashboard: Settings panel
- Engine: `--db-path` argument

---

## 📊 Dashboard Features

### Active Trade Tab
- Real-time P&L tracking
- Entry price, current price, TG/SL levels
- Hold time countdown
- Manual exit / Cancel pending buttons

### History Tab
- Load trades from SQLite database
- Date range filtering (IST)
- Delete old trades by date range
- Export to Excel/CSV
- Color-coded P&L (green/red)

### Logs Tab
- Real-time feed timestamps (IST)
- Connection status
- Pattern detections
- Trade entries/exits

---

## 🔧 Troubleshooting

### Database Connection Error
- Verify DB path: `G:\Projects\Centralize Data Centre\data\nifty_local.db`
- Ensure data collector is running (during market hours)
- Check file permissions

### WebSocket Connection Issues
- Verify Angel One credentials in `.env`
- Check internet connection
- Engine will auto-reconnect on disconnect

### No Trades Being Taken
- Check if market is open (collector only runs during market hours)
- Verify trend signals are being detected (check logs)
- Ensure pattern thresholds are met

---

## 📈 Understanding the System

### Hybrid Architecture

1. **Signal Generation** (DB Polling):
   - Polls SQLite DB every 5 seconds for new NIFTY 50 ticks
   - Builds 30-second candles
   - Detects trend signals (0.11% cumulative move)

2. **Pattern Analysis** (DB Query):
   - Queries last 60 seconds of option data
   - Calculates fast indicators (IV, Delta, Volume, Premium)
   - Generates trade signal with direction

3. **Trade Entry** (DB LTP):
   - Uses LTP from database at/after signal time
   - Selects ATM option (CE/PE) based on direction

4. **Trade Exit** (WebSocket):
   - Subscribes to Angel One WebSocket for active option
   - Monitors real-time LTP for target/stop loss
   - Time-based exit after 3 minutes

### Real-time Safety

- All decisions based on strictly past data (no lookahead)
- Timestamps from broadcaster/DB (not system time)
- Single active trade enforcement

---

## 📝 Requirements

### Python Packages

Install from `requirements_dashboard.txt`:
```powershell
py -m pip install -r requirements_dashboard.txt
```

For VPS data collector:
```powershell
cd vps_data_collector
py -m pip install -r requirements.txt
```

---

## 📁 Archive

Old/unused files have been moved to `Archive/`:
- `Archive/old_scripts/`: Old analysis and backtest scripts
- `Archive/old_results/`: Old CSV results and trade logs
- `Archive/old_logs/`: Old log files and trade charts
- `Archive/old_docs/`: Old documentation
- `Archive/old_versions/`: Previous versions

---

## ✅ Project Status

- ✅ Live paper trading system (dashboard + engine)
- ✅ Real-time trend detection
- ✅ Pattern analysis with fast indicators
- ✅ Angel One WebSocket integration
- ✅ SQLite trade history
- ✅ NiceGUI dashboard with export
- ✅ Single active trade enforcement
- ✅ IST timestamp handling

**Ready for git push** 🚀
