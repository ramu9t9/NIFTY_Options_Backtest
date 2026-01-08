# Startup Scripts

This folder contains all the startup scripts for the NIFTY Options Paper Trading System.

---

## Quick Start

### Live Mode (Real-time Trading)
```batch
# Step 1: Start data collector
scripts\start_live_data_collector.bat

# Step 2: Start paper trading engine
scripts\start_live_paper_trading.bat
```

### Replay Mode (Historical Testing)
```batch
# Step 1: Start broadcaster data writer
scripts\start_broadcaster_writer.bat

# Step 2: Start dashboard
scripts\start_live_dashboard.bat
```

---

## Script Descriptions

### Live Mode Scripts

#### `start_live_data_collector.bat`
- **Purpose**: Starts Angel One data collector
- **Database**: Writes to `data/nifty_live.db`
- **Frequency**: Collects data every 5 seconds
- **Use**: Run during market hours (9:15 AM - 3:30 PM IST)

#### `start_live_paper_trading.bat`
- **Purpose**: Starts paper trading engine (database polling)
- **Database**: Reads from `data/nifty_live.db`
- **Output**: Creates CSV file with trade results
- **Use**: Run after starting data collector

---

### Replay Mode Scripts

#### `start_broadcaster_writer.bat`
- **Purpose**: Receives data from broadcaster and writes to database
- **Source**: `ws://localhost:8765` (Centralize Data Centre broadcaster)
- **Database**: Writes to `data/nifty_replay.db`
- **Use**: Run when broadcaster is active

#### `start_replay_paper_trading.bat`
- **Purpose**: Starts paper trading engine for replay mode
- **Database**: Reads from `data/nifty_replay.db`
- **Output**: Creates CSV file with trade results
- **Use**: Alternative to dashboard for replay mode

#### `start_live_dashboard.bat`
- **Purpose**: Starts NiceGUI web dashboard
- **URL**: http://localhost:8080
- **Features**: Real-time monitoring, trade management, export
- **Best For**: Replay mode with broadcaster feed

---

## Recommended Workflow

### For Live Trading
1. Open Terminal 1: `scripts\start_live_data_collector.bat`
2. Wait for "✅ Successfully wrote 24 records"
3. Open Terminal 2: `scripts\start_live_paper_trading.bat`
4. Monitor console for trade signals

### For Replay Testing
1. Ensure broadcaster is running at `ws://localhost:8765`
2. Open Terminal 1: `scripts\start_broadcaster_writer.bat`
3. Open Terminal 2: `scripts\start_live_dashboard.bat`
4. Open browser: http://localhost:8080
5. Select "Replay Mode" and click "Connect + Start Paper Trading"

---

## Troubleshooting

### Data Collector Won't Start
- Check Angel One credentials in `vps_data_collector/nifty_stream_local_sqlite.py`
- Verify market hours (9:15 AM - 3:30 PM IST, Mon-Fri)

### Paper Trading Engine Shows No Data
- Ensure data collector is running and writing to database
- Check database exists: `data/nifty_live.db` or `data/nifty_replay.db`
- Verify database has records (should grow every 5 seconds)

### Dashboard Won't Connect
- Ensure broadcaster is running at `ws://localhost:8765`
- Check if broadcaster writer is receiving data
- Verify NiceGUI is installed: `pip install nicegui`

---

## Database Locations

- **Live Mode**: `g:\Projects\NIFTY_Options_Backtest\data\nifty_live.db`
- **Replay Mode**: `g:\Projects\NIFTY_Options_Backtest\data\nifty_replay.db`
- **Trade History**: `g:\Projects\NIFTY_Options_Backtest\exports\live_trades.db`

---

## Output Files

Trade results are saved to:
- **Paper Trading Engine**: `paper_trading/paper_trades_YYYYMMDD_HHMMSS.csv`
- **Dashboard Exports**: `exports/dashboard_trades_YYYYMMDD_HHMMSS.csv` or `.xlsx`
