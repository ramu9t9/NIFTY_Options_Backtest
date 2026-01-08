# Quick Reference Card

## 🎯 How to Start the System

### Option 1: Live Mode (Real Market Data)
1. Open Command Prompt
2. Navigate to project: `cd g:\Projects\NIFTY_Options_Backtest`
3. Run: `scripts\start_live_data_collector.bat` (Terminal 1)
4. Run: `scripts\start_live_paper_trading.bat` (Terminal 2)

### Option 2: Replay Mode (Historical Data)
1. Open Command Prompt
2. Navigate to project: `cd g:\Projects\NIFTY_Options_Backtest`
3. Run: `scripts\start_broadcaster_writer.bat` (Terminal 1)
4. Run: `scripts\start_live_dashboard.bat` (Terminal 2)
5. Open browser: http://localhost:8080

---

## 📂 Project Organization

```
NIFTY_Options_Backtest/
│
├── 📁 scripts/              ⭐ START HERE - All startup scripts
│   ├── start_live_data_collector.bat
│   ├── start_live_paper_trading.bat
│   ├── start_broadcaster_writer.bat
│   ├── start_replay_paper_trading.bat
│   ├── start_live_dashboard.bat
│   └── README.md           📖 Detailed script documentation
│
├── 📁 data/                 💾 Market data databases
│   ├── nifty_live.db
│   └── nifty_replay.db
│
├── 📁 paper_trading/        🤖 Trading engine
├── 📁 vps_data_collector/   📡 Data collection
├── 📁 exports/              📊 Trade results
│
├── live_dashboard.py        🖥️ Web dashboard
├── broadcaster_data_writer.py
└── README.md               📚 Full documentation
```

---

## 🔧 Common Tasks

### View Live Trades
- Check console output in Terminal 2
- Or open: `paper_trading/paper_trades_*.csv`

### View Dashboard
- Start: `scripts\start_live_dashboard.bat`
- Open: http://localhost:8080

### Check Database
- Live: `data\nifty_live.db`
- Replay: `data\nifty_replay.db`

### Export Trades
- Dashboard: Click "Export CSV" or "Export Excel"
- Files saved to: `exports/`

---

## ❓ Troubleshooting

### No data in database?
- Ensure data collector is running
- Check market hours (9:15 AM - 3:30 PM IST)

### Paper trading not finding data?
- Verify database exists and has records
- Check database path in script

### Dashboard won't connect?
- Ensure broadcaster is running
- Check broadcaster writer is receiving data

---

## 📞 Need Help?

1. Check `scripts/README.md` for detailed instructions
2. Check main `README.md` for full documentation
3. Check `walkthrough.md` in `.gemini` folder for implementation details
