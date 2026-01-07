# 📡 How to Use WebSocket Broadcaster - Simple Guide

## 🎯 Quick Overview

**Problem**: Angel One API allows only 2-4 concurrent logins → Multiple projects = Login conflicts ❌

**Solution**: One project logs in, broadcasts data to all others ✅

---

## 🏗️ Architecture

```
================================================================================
                    WEBSOCKET BROADCASTER ARCHITECTURE
================================================================================

┌────────────────────┐
│   ANGEL ONE API    │  (External API - Login Limit: 2-4)
└─────────┬──────────┘
          │
          │ API Login + Data Fetch
          │
          ▼
┌──────────────────────────────────────────────┐
│  CENTRALIZE DATA CENTRE PROJECT              │
│  (This project logs into Angel One)          │
│                                              │
│  1. Data Collector                           │
│     • Logs into Angel One                    │
│     • Fetches: Spot, Futures, Options, IV,   │
│       Greeks (Delta, Gamma, Theta, Vega)     │
│     • Stores in database                     │
│                                              │
│  2. Database (nifty_local.db)                │
│     • Stores all market data                 │
│                                              │
│  3. WebSocket Broadcaster                    │
│     • Reads from database                    │
│     • Broadcasts to all clients              │
│     • URL: ws://localhost:8765               │
│     • Updates every 5 seconds                │
│                                              │
└───────────┬──────────────────────────────────┘
            │
            │ WebSocket Connection
            │ (No login needed - unlimited connections)
            │
    ┌───────┴───────┬───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ PROJECT │    │ PROJECT │    │ PROJECT │
│   A     │    │   B     │    │   C     │
│         │    │         │    │         │
│ NO      │    │ NO      │    │ NO      │
│ LOGIN   │    │ LOGIN   │    │ LOGIN   │
│ NEEDED  │    │ NEEDED  │    │ NEEDED  │
└─────────┘    └─────────┘    └─────────┘

================================================================================
```

---

## ✅ Key Points

### 1. **ONLY "Centralize Data Centre" logs into Angel One**
   - Other projects **DON'T need** Angel One credentials
   - No API keys, user IDs, or passwords needed in your project

### 2. **Your project connects to WebSocket broadcaster**
   - URL: `ws://localhost:8765`
   - No login required
   - Unlimited projects can connect simultaneously

### 3. **Data Flow:**
   ```
   Angel One API 
   → Centralize Data Centre 
   → Database 
   → Broadcaster 
   → Your Project
   ```

### 4. **Benefits:**
   - ✅ No login conflicts (only 1 login total)
   - ✅ Unlimited projects can connect
   - ✅ Real-time data (updated every 5 seconds)
   - ✅ No API rate limits from your side

---

## 🚀 Setup Steps

### Step 1: Start Centralize Data Centre

```bash
cd "G:\Projects\Centralize Data Centre"
.\start_all_services.bat
```

This starts:
- Data Collector (logs into Angel One)
- WebSocket Broadcaster (broadcasts data)

**Verify it's running:**
- Check: `ws://localhost:8765` should be accessible
- Or run: `.\check_services_status.bat`

### Step 2: Install Required Library (One Time)

```bash
pip install websocket-client
```

**Important**: Use `websocket-client` (not `websockets`)

### Step 3: Use in Your Project

The system is **already configured** to use WebSocket! Just start your components:

```bash
# Start Trading Engine
py trading_engine.py

# Start Dashboard
py dashboard_ui.py
```

Both will automatically connect to `ws://localhost:8765` and receive data.

---

## 📋 What Your Project Does

### ❌ DO NOT:
- Log into Angel One API
- Make direct API calls
- Need Angel One credentials
- Start Data Collector (not needed)

### ✅ DO:
- Connect to WebSocket: `ws://localhost:8765`
- Receive real-time data automatically
- Use data for trading decisions
- Start Trading Engine and Dashboard

---

## 🔍 How It Works in Your Project

### Trading Engine (`trading_engine.py`)
- Uses `WebSocketDataAdapter` (automatically)
- Connects to broadcaster on startup
- Receives data every 5 seconds
- No API login needed

### Dashboard (`dashboard_ui.py`)
- Uses `WebSocketDataAdapter` (automatically)
- Shows WebSocket connection status
- Displays real-time data
- No API login needed

### Data Flow in Your Project:
```
WebSocket Broadcaster (ws://localhost:8765)
    ↓
WebSocketDataReader (connects automatically)
    ↓
WebSocketDataAdapter (compatible interface)
    ↓
Trading Engine / Dashboard (uses data)
```

---

## ⚙️ Configuration

### WebSocket URL
Default: `ws://localhost:8765`

If your broadcaster runs on a different port, update in:
- `src/websocket_data_reader.py` (line 52)
- `src/websocket_data_adapter.py` (line 20)

### Connection Timeout
- Default wait: 10-15 seconds for initial connection
- Data arrives every 5 seconds after connection

---

## 🧪 Testing

### Verify WebSocket Connection

Run the verification script:
```bash
py verify_websocket_data.py
```

This checks:
- ✓ WebSocket connection
- ✓ Spot price (NIFTY 50)
- ✓ Options data
- ✓ Futures price
- ✓ IV (Implied Volatility)
- ✓ Greeks (Delta, Gamma, Theta, Vega)
- ✓ Data freshness
- ✓ Cache functionality

### Expected Output:
```
✓ All verifications passed! WebSocket integration is working correctly.
```

---

## 📊 Dashboard Status

In the **Controls** tab, you'll see:

### 📡 WebSocket Data Source
- **Status**: CONNECTED / DISCONNECTED
- **Symbols**: Number of symbols received
- **Messages**: Total messages received
- **URL**: ws://localhost:8765

### Status Meanings:
- **CONNECTED** (Green): WebSocket connected, data flowing
- **CONNECTED (NO DATA)** (Orange): Connected but no data yet (wait 5-10 seconds)
- **DISCONNECTED** (Red): Not connected (check if broadcaster is running)

---

## 🐛 Troubleshooting

### Issue: "DISCONNECTED" Status

**Solution:**
1. Check if Centralize Data Centre is running
2. Verify broadcaster is on port 8765
3. Check firewall settings
4. Wait 10-15 seconds after starting

### Issue: "CONNECTED (NO DATA)"

**Solution:**
1. Wait 5-10 seconds (data arrives every 5 seconds)
2. Check if Data Collector in Centralize Data Centre is running
3. Verify database has recent data

### Issue: Connection Refused

**Solution:**
1. Start Centralize Data Centre: `.\start_all_services.bat`
2. Check port 8765: `netstat -an | findstr ":8765"`
3. Verify broadcaster logs for errors

---

## 📝 Code Example

If you want to use WebSocket directly in your code:

```python
from src.websocket_data_reader import WebSocketDataReader
import time

# Initialize (auto-connects)
reader = WebSocketDataReader(ws_url="ws://localhost:8765")

# Wait for data (first batch takes ~5 seconds)
time.sleep(6)

# Get spot price
spot = reader.get_spot_ltp("NIFTY 50")
print(f"NIFTY 50: {spot}")

# Get watchlist
symbols = reader.pick_watchlist(atm_window=5)

# Fetch market data
market_data = reader.fetch_market_data(symbols)

# Fetch Greeks
greeks_data = reader.fetch_greeks_data("NIFTY 50")
```

---

## 🎯 Summary

1. **Start Centralize Data Centre** (separate project)
2. **Your project automatically connects** to WebSocket
3. **No API login needed** in your project
4. **Data flows automatically** every 5 seconds
5. **Use data** in Trading Engine and Dashboard

**That's it!** The system handles everything automatically.

---

## 📚 Related Files

- `src/websocket_data_reader.py` - Core WebSocket client
- `src/websocket_data_adapter.py` - Adapter for compatibility
- `verify_websocket_data.py` - Verification script
- `WEBSOCKET_INTEGRATION_README.md` - Detailed integration guide
- `WEBSOCKET_MIGRATION_COMPLETE.md` - Migration status

---

**Last Updated**: January 2, 2026  
**Status**: Production Ready ✅

