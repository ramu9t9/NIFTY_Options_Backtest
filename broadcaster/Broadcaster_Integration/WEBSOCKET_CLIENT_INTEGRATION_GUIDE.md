# 📡 WebSocket Client Integration Guide

## Overview

This guide explains how to connect to the **Centralize Data Centre** WebSocket broadcaster and receive real-time NIFTY options data in your projects.

## 🎯 What You Get

- **Real-time data** every 5 seconds
- **All fields**: LTP, Volume, OI, Greeks (Delta, Gamma, Theta, Vega), IV
- **Multiple symbols**: NIFTY 50, Futures, Options (ATM ±5 strikes)
- **Low latency**: <100ms from database to your application
- **Unlimited connections**: Multiple projects can connect simultaneously

---

## 🚀 Quick Start

### Prerequisites

1. **WebSocket Broadcaster must be running**
   - Check status: `.\check_services_status.bat`
   - Start if needed: `.\start_all_services.bat`

2. **WebSocket URL**: `ws://localhost:8765`

3. **Python dependencies** (if using Python):
   ```bash
   pip install websocket-client
   ```
   
   **Important**: Install `websocket-client` (not `websockets`). The broadcaster uses the `websocket-client` library.

---

## 📋 Method 1: Using WebSocketDataReader (Recommended for AI Trading System)

### Python Projects - Full Integration

If you're integrating with the AI Trading Assistant system, use the `WebSocketDataReader` class which provides the same interface as `OIDataCollector`:

```python
from app.data.websocket_data_reader import WebSocketDataReader
from app.core.config import settings

# Initialize WebSocket reader
reader = WebSocketDataReader(ws_url="ws://localhost:8765")

# Wait for connection (automatic, but you can check)
import time
for i in range(10):
    if reader.is_connected():
        break
    time.sleep(0.5)

# Get spot price
spot = reader.get_spot_ltp("NIFTY 50")
print(f"NIFTY 50 Spot: {spot}")

# Get watchlist (ATM ±5 strikes)
symbols = reader.pick_watchlist(atm_window=5, index_symbol="NIFTY 50")
print(f"Watchlist: {len(symbols)} symbols")

# Fetch market data
market_data = reader.fetch_market_data(symbols)
print(f"Market data for {len(market_data)} symbols")

# Fetch Greeks
greeks_data = reader.fetch_greeks_data("NIFTY 50")
print(f"Greeks for {len(greeks_data)} options")

# Collect complete data (all at once)
symbols, market_data, greeks_data = reader.collect_complete_data(
    atm_window=5, 
    index_symbol="NIFTY 50"
)
```

### Using the Simple Client Library

Alternatively, you can use the provided `realtime_client.py` library for basic usage:

```python
from services.realtime_client import RealtimeDataClient

# Define callback for real-time data
def on_data(data):
    """Called whenever new data arrives"""
    symbol = data.get('symbol')
    ltp = data.get('ltp')
    volume = data.get('volume')
    delta = data.get('delta')
    iv = data.get('iv')
    
    print(f"{symbol}: LTP={ltp}, Volume={volume}, Delta={delta}, IV={iv}%")

# Create client
client = RealtimeDataClient(
    ws_url="ws://localhost:8765",
    on_data_callback=on_data
)

# Connect to real-time feed
client.connect()

# Query historical data
latest = client.get_latest("NIFTY 50")
print(f"Latest NIFTY 50: {latest}")

# Get all symbols
symbols = client.get_symbols()
print(f"Available symbols: {len(symbols)}")

# Keep running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    client.close()
```

### Available Methods

```python
# Get latest tick for a symbol
latest = client.get_latest("NIFTY06JAN2626500CE")

# Get latest ticks for all symbols
all_latest = client.get_latest_all(limit=100)

# Get historical data
historical = client.get_historical(
    symbol="NIFTY 50",
    start_ts="2026-01-02T08:00:00+00:00",
    end_ts="2026-01-02T09:00:00+00:00"
)

# Get all available symbols
symbols = client.get_symbols()

# Custom SQL query
results = client.query(
    "SELECT * FROM ltp_ticks WHERE symbol = ? ORDER BY ts DESC LIMIT 10",
    ("NIFTY 50",)
)
```

---

## 📋 Method 2: Direct WebSocket Connection

### Python Example (Using websocket-client)

**Note**: The broadcaster uses `websocket-client` (synchronous), not `websockets` (async).

```python
import websocket
import json
import threading
import time

def on_message(ws, message):
    """Handle incoming messages"""
    try:
        data = json.loads(message)
        
        # Skip welcome/ping messages
        if data.get('type') in ['welcome', 'pong']:
            if data.get('type') == 'welcome':
                print(f"Connected: {data.get('message')}")
            return
        
        # Process real-time data
        symbol = data.get('symbol')
        ltp = data.get('ltp')
        volume = data.get('volume')
        delta = data.get('delta')
        iv = data.get('iv')
        
        print(f"{symbol}: LTP={ltp}, Volume={volume}, Delta={delta}, IV={iv}%")
    except json.JSONDecodeError as e:
        print(f"Error decoding message: {e}")

def on_error(ws, error):
    """Handle errors"""
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """Handle connection close"""
    print("WebSocket connection closed")

def on_open(ws):
    """Handle connection open"""
    print("WebSocket connection established")

# Connect to broadcaster
ws_url = "ws://localhost:8765"
ws = websocket.WebSocketApp(
    ws_url,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open
)

# Run in separate thread
ws_thread = threading.Thread(target=ws.run_forever)
ws_thread.daemon = True
ws_thread.start()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    ws.close()
    print("Disconnected")
```

### JavaScript/Node.js Example

```javascript
const WebSocket = require('ws');

// Connect to WebSocket broadcaster
const ws = new WebSocket('ws://localhost:8765');

ws.on('open', function open() {
    console.log('Connected to WebSocket broadcaster');
});

ws.on('message', function incoming(data) {
    const message = JSON.parse(data);
    
    // Skip welcome/ping messages
    if (message.type === 'welcome' || message.type === 'pong') {
        return;
    }
    
    // Process real-time data
    const symbol = message.symbol;
    const ltp = message.ltp;
    const volume = message.volume;
    const delta = message.delta;
    const iv = message.iv;
    
    console.log(`${symbol}: LTP=${ltp}, Volume=${volume}, Delta=${delta}, IV=${iv}%`);
});

ws.on('error', function error(err) {
    console.error('WebSocket error:', err);
    console.log('Make sure WebSocket broadcaster is running!');
});

ws.on('close', function close() {
    console.log('Disconnected from WebSocket broadcaster');
});
```

### C# Example

```csharp
using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

class WebSocketClient
{
    static async Task Main(string[] args)
    {
        using (var ws = new ClientWebSocket())
        {
            try
            {
                await ws.ConnectAsync(new Uri("ws://localhost:8765"), CancellationToken.None);
                Console.WriteLine("Connected to WebSocket broadcaster");
                
                var buffer = new byte[1024 * 4];
                
                while (ws.State == WebSocketState.Open)
                {
                    var result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                    
                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        var message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        var data = JsonConvert.DeserializeObject<dynamic>(message);
                        
                        // Skip welcome/ping messages
                        if (data.type == "welcome" || data.type == "pong")
                            continue;
                        
                        // Process data
                        Console.WriteLine($"{data.symbol}: LTP={data.ltp}, Volume={data.volume}, Delta={data.delta}, IV={data.iv}%");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine("Make sure WebSocket broadcaster is running!");
            }
        }
    }
}
```

---

## 📊 Data Format

### Message Structure

Each message is a JSON object with the following structure:

```json
{
  "symbol": "NIFTY06JAN2626500CE",
  "token": "49873",
  "ts": "2026-01-02T08:13:05+00:00",
  "ltp": 12.0,
  "bid": 11.95,
  "ask": 12.05,
  "volume": 161816655,
  "oi": 12911015,
  "delta": 0.4115,
  "gamma": 0.0005,
  "theta": -35.21,
  "vega": 10.64,
  "iv": 5.66,
  "source": "api"
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Trading symbol (e.g., "NIFTY 50", "NIFTY06JAN2626500CE") |
| `token` | string | Angel One token ID |
| `ts` | string | Timestamp (ISO 8601 format, UTC) |
| `ltp` | float | Last Traded Price |
| `bid` | float | Best Bid Price |
| `ask` | float | Best Ask Price |
| `volume` | integer | Trading Volume (0 for indices) |
| `oi` | integer | Open Interest (NULL for indices) |
| `delta` | float | Option Delta (NULL for non-options) |
| `gamma` | float | Option Gamma (NULL for non-options) |
| `theta` | float | Option Theta (NULL for non-options) |
| `vega` | float | Option Vega (NULL for non-options) |
| `iv` | float | Implied Volatility % (NULL for non-options) |
| `source` | string | Data source ("api", "ws", "local_ws") |

### Special Messages

#### Welcome Message
```json
{
  "type": "welcome",
  "message": "Connected to Real-Time Data Centre",
  "timestamp": "2026-01-02T08:13:05+00:00",
  "subscribers": 1
}
```

#### Ping/Pong
```json
{
  "type": "pong",
  "timestamp": "2026-01-02T08:13:05+00:00"
}
```

---

## 🔧 Advanced Usage

### Request Latest Data for a Symbol

You can send requests to the broadcaster:

```python
import asyncio
import websockets
import json

async def request_latest():
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # Wait for welcome
        await websocket.recv()
        
        # Request latest data for a symbol
        request = {
            "type": "get_latest",
            "symbol": "NIFTY 50"
        }
        await websocket.send(json.dumps(request))
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Latest NIFTY 50: {data}")

asyncio.run(request_latest())
```

### Ping/Pong (Keep-Alive)

```python
# Send ping to keep connection alive
request = {"type": "ping"}
await websocket.send(json.dumps(request))

# Server responds with pong
response = await websocket.recv()
```

---

## 🗄️ Direct Database Access

If you need historical data or want to query the database directly:

### Python Example

```python
import sqlite3
from pathlib import Path

# Database path
db_path = Path(r"G:\Projects\Centralize Data Centre\data\nifty_local.db")

# Connect to database
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Query latest data for a symbol
cursor.execute("""
    SELECT * FROM ltp_ticks 
    WHERE symbol = ? 
    ORDER BY ts DESC 
    LIMIT 1
""", ("NIFTY 50",))

row = cursor.fetchone()
if row:
    print(f"Latest: {dict(row)}")

# Query historical data
cursor.execute("""
    SELECT * FROM ltp_ticks 
    WHERE symbol = ? AND ts >= ? AND ts <= ?
    ORDER BY ts
""", ("NIFTY 50", "2026-01-02T08:00:00+00:00", "2026-01-02T09:00:00+00:00"))

for row in cursor.fetchall():
    print(dict(row))

conn.close()
```

**Note**: For database schema details, see [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md)

---

## ⚠️ Error Handling

### Connection Errors

```python
try:
    async with websockets.connect("ws://localhost:8765") as websocket:
        # ... your code ...
except ConnectionRefusedError:
    print("Error: WebSocket broadcaster not running!")
    print("Start it with: .\\start_all_services.bat")
except websockets.exceptions.ConnectionClosed:
    print("Connection closed. Reconnecting...")
    # Implement reconnection logic
except Exception as e:
    print(f"Error: {e}")
```

### Reconnection Logic

```python
import asyncio
import websockets

async def connect_with_retry(max_retries=5, delay=5):
    """Connect with automatic retry"""
    for attempt in range(max_retries):
        try:
            async with websockets.connect("ws://localhost:8765") as websocket:
                print("Connected!")
                async for message in websocket:
                    # Process messages
                    pass
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                print(f"Connection refused. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print("Max retries reached. Please start the broadcaster.")
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(delay)

asyncio.run(connect_with_retry())
```

---

## 📈 Best Practices

### 1. Wait for Data After Connection

WebSocket connection is immediate, but data arrives every 5 seconds:

```python
reader = WebSocketDataReader()
# Connection is established immediately
# But wait for first data batch
import time
time.sleep(6)  # Wait slightly longer than broadcast interval

# Now check cache
print(f"Symbols available: {len(reader._data_cache)}")
```

### 2. Handle Futures Expiry Mismatch

Futures often have different expiry than options:

```python
# Options: NIFTY06JAN26{STRIKE}CE
# Futures: NIFTY27JAN26FUT (different expiry!)

# When looking for futures, search by index base, not expiry
index_base = "NIFTY"  # or "BANKNIFTY"
for symbol in reader._data_cache.keys():
    if "FUT" in symbol and index_base in symbol:
        # Found future (regardless of expiry)
        fut_data = reader._data_cache[symbol]
        break
```

### 3. Filter Data by Symbol

Only process data for symbols you need:

```python
def on_data(data):
    symbol = data.get('symbol')
    
    # Only process NIFTY 50 and specific options
    if symbol == "NIFTY 50" or "26500" in symbol:
        # Process data
        process_data(data)
```

### 2. Buffer and Batch Process

Collect data and process in batches:

```python
buffer = []

def on_data(data):
    buffer.append(data)
    
    # Process every 10 records
    if len(buffer) >= 10:
        process_batch(buffer)
        buffer.clear()
```

### 3. Handle Missing Fields

Some fields may be NULL for certain symbol types:

```python
def on_data(data):
    # Check if it's an option (has Greeks)
    if data.get('delta') is not None:
        # It's an option
        delta = data.get('delta')
        iv = data.get('iv')
        # Process option data
    else:
        # It's an index or future
        ltp = data.get('ltp')
        # Process index/future data
```

### 4. Use Threading for Heavy Processing

```python
import threading

def process_data_async(data):
    """Process data in background thread"""
    # Heavy processing here
    pass

def on_data(data):
    # Process in background to avoid blocking
    thread = threading.Thread(target=process_data_async, args=(data,))
    thread.start()
```

### 5. Log Important Events

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def on_data(data):
    try:
        # Process data
        process_data(data)
    except Exception as e:
        logger.error(f"Error processing {data.get('symbol')}: {e}")
```

---

## 🔍 Troubleshooting

### Issue: Connection Refused

**Solution:**
1. Check if WebSocket Broadcaster is running:
   ```powershell
   .\check_services_status.bat
   ```
2. Start the broadcaster:
   ```powershell
   .\start_all_services.bat
   ```
3. Verify port 8765 is listening:
   ```powershell
   netstat -an | findstr ":8765"
   ```

### Issue: No Data Received

**Possible causes:**
1. VPS Data Collector not running
2. Database empty or not being updated
3. WebSocket connection established but no new data

**Solution:**
1. Check VPS collector is running
2. Verify database has recent data:
   ```python
   import sqlite3
   conn = sqlite3.connect(r"G:\Projects\Centralize Data Centre\data\nifty_local.db")
   cursor = conn.cursor()
   cursor.execute("SELECT MAX(ts) FROM ltp_ticks")
   print(f"Latest data: {cursor.fetchone()[0]}")
   ```

### Issue: Data Fields are NULL

**Explanation:**
- Indices (NIFTY 50) don't have Volume, OI, Greeks, or IV
- Futures don't have Greeks or IV
- Only Options have all fields

**Solution:**
- Check symbol type before accessing fields
- Use conditional checks: `if data.get('delta') is not None:`

---

## 📚 Additional Resources

- **Database Schema**: See [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md)
- **Architecture**: See [ARCHITECTURE_IMPLEMENTATION_PLAN.md](ARCHITECTURE_IMPLEMENTATION_PLAN.md)
- **Quick Start**: See [QUICK_START.md](QUICK_START.md)

---

## 💡 Example Projects

### Simple Price Monitor

```python
from services.realtime_client import RealtimeDataClient

prices = {}

def on_data(data):
    symbol = data.get('symbol')
    ltp = data.get('ltp')
    prices[symbol] = ltp
    print(f"{symbol}: {ltp}")

client = RealtimeDataClient(on_data_callback=on_data)
client.connect()

# Keep running
import time
while True:
    time.sleep(1)
```

### Options Greeks Monitor

```python
from services.realtime_client import RealtimeDataClient

def on_data(data):
    # Only process options (have Greeks)
    if data.get('delta') is not None:
        symbol = data.get('symbol')
        delta = data.get('delta')
        gamma = data.get('gamma')
        theta = data.get('theta')
        vega = data.get('vega')
        iv = data.get('iv')
        
        print(f"{symbol}:")
        print(f"  Delta: {delta:.4f}, Gamma: {gamma:.6f}")
        print(f"  Theta: {theta:.2f}, Vega: {vega:.2f}")
        print(f"  IV: {iv:.2f}%")

client = RealtimeDataClient(on_data_callback=on_data)
client.connect()

import time
while True:
    time.sleep(1)
```

### Volume Alert System

```python
from services.realtime_client import RealtimeDataClient

def on_data(data):
    symbol = data.get('symbol')
    volume = data.get('volume', 0)
    
    # Alert if volume exceeds threshold
    if volume > 10000000:  # 10 million
        print(f"ALERT: {symbol} volume spike: {volume:,}")

client = RealtimeDataClient(on_data_callback=on_data)
client.connect()

import time
while True:
    time.sleep(1)
```

---

## 🔧 Integration with AI Trading System

### Complete Integration Example

If you're integrating WebSocket data reader into the AI Trading Assistant system:

```python
from app.core.config import settings
from app.domain.market.snapshot_builder import SnapshotBuilder
from app.data.websocket_data_reader import WebSocketDataReader

# Set data source mode to websocket (in .env or code)
settings.DATA_SOURCE_MODE = "websocket"
settings.WEBSOCKET_URL = "ws://localhost:8765"

# Initialize (no angel_client needed!)
snapshot_builder = SnapshotBuilder(angel_client=None, storage=storage)

# Build snapshot - automatically uses WebSocket
snapshot = snapshot_builder.build_snapshot(index_symbol="NIFTY 50")
```

### Configuration

Add to your `.env` file:

```bash
DATA_SOURCE_MODE=websocket
WEBSOCKET_URL=ws://localhost:8765
```

The `SnapshotBuilder` will automatically:
- Detect `DATA_SOURCE_MODE=websocket`
- Initialize `WebSocketDataReader` instead of `OIDataCollector`
- Skip Angel One API login
- Use WebSocket broadcaster for all data

### Verification

Use the verification script to check data reception:

```bash
cd "momentum_scalper_ai\ai_trading_assistant"
python verify_data_reception.py
```

This verifies:
- Spot price
- Futures price
- Options data
- IV (Implied Volatility)
- Greeks (Delta, Gamma, Theta, Vega)
- Data freshness
- Collection rate

---

## 📝 Implementation Notes (From Real-World Integration)

### Key Learnings

1. **Futures Expiry Mismatch**: 
   - Options and futures often have different expiries
   - Always search for FUT symbols by index base, not expiry
   - Example: Options `06JAN26`, Futures `27JAN26`
   - Solution: Search for any FUT symbol containing index base (e.g., "NIFTY")

2. **Data Arrival Timing**:
   - Connection is immediate, but first data batch takes ~5 seconds
   - Always wait after connection before accessing data
   - Check `is_connected()` and cache size before using data
   - Recommended: Wait 5-6 seconds after connection

3. **Symbol Format**:
   - Options: `NIFTY{EXPIRY}{STRIKE}CE` or `...PE` (e.g., `NIFTY06JAN2626050CE`)
   - Futures: `NIFTY{EXPIRY}FUT` (e.g., `NIFTY27JAN26FUT`)
   - Index: `NIFTY 50` or `NIFTY BANK`

4. **Cache Management**:
   - Data is cached in `_data_cache` dictionary (in WebSocketDataReader)
   - Key: symbol string, Value: data dict
   - Cache is updated every 5 seconds with new broadcast
   - No need to manually refresh - automatic updates
   - Use thread-safe access when reading from cache

5. **Error Handling**:
   - Connection failures: Check broadcaster status
   - No data: Wait longer, verify broadcaster is running
   - Missing fields: Check symbol type (indices/futures don't have all fields)
   - Import errors: Ensure `websocket-client` is installed (not `websockets`)

6. **Library Confusion**:
   - Use `websocket-client` (synchronous, used by broadcaster)
   - NOT `websockets` (async library, different package)
   - Install: `pip install websocket-client`

---

## 🎯 Quick Reference

### Installation
```bash
pip install websocket-client  # Note: websocket-client, not websockets
```

### Basic Usage (WebSocketDataReader)
```python
from app.data.websocket_data_reader import WebSocketDataReader

reader = WebSocketDataReader(ws_url="ws://localhost:8765")
import time
time.sleep(5)  # Wait for data

spot = reader.get_spot_ltp("NIFTY 50")
symbols, market_data, greeks = reader.collect_complete_data()
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| No data received | Wait 5-10 seconds after connection |
| Future price missing | Futures have different expiry - search by index base |
| Import error | Install `websocket-client` (not `websockets`) |
| Connection refused | Start broadcaster: `.\start_all_services.bat` |
| Missing Greeks | Wait for next broadcast cycle (5 seconds) |
| Cache empty | Wait longer, verify broadcaster is broadcasting |

---

## 📞 Support

If you encounter issues:

1. Check service status: `.\check_services_status.bat`
2. Check logs: `data\broadcaster_service.log`
3. Verify database: See [DATABASE_SCHEMA_REFERENCE.md](DATABASE_SCHEMA_REFERENCE.md)
4. Review architecture: See [ARCHITECTURE_IMPLEMENTATION_PLAN.md](ARCHITECTURE_IMPLEMENTATION_PLAN.md)
5. Run verification script: `python verify_data_reception.py` (for AI Trading System)

---

**Last Updated:** January 2, 2026  
**Version:** 1.1.0

**Changes in v1.1.0:**
- Added WebSocketDataReader integration guide
- Added futures expiry mismatch handling
- Added verification script documentation
- Added implementation notes from real-world integration
- Fixed library name (websocket-client, not websockets)
- Added complete troubleshooting section with real-world issues
- Added quick reference table

