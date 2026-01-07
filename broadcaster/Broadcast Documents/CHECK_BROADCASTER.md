# ⚠️ WebSocket Broadcaster Connection Issue

## Problem
The WebSocket connection to `ws://localhost:8765` is not opening. The `on_open` callback is never called.

## Diagnosis
- Port 8765 test: **FAILED** (`TcpTestSucceeded: False`)
- WebSocket connection: **NOT OPENING**
- Thread status: **ALIVE** (but not connecting)

## Possible Causes

1. **Broadcaster Not Running**
   - The broadcaster may have stopped
   - Check: `netstat -an | findstr ":8765"` should show LISTENING

2. **IPv4 vs IPv6 Issue**
   - Broadcaster might be listening on IPv6 only
   - Try: `ws://[::1]:8765` or `ws://127.0.0.1:8765`

3. **Firewall Blocking**
   - Windows Firewall might be blocking the connection
   - Check firewall settings

4. **Broadcaster Configuration**
   - Broadcaster might need to be restarted
   - Check broadcaster logs for errors

## Solution Steps

1. **Verify Broadcaster is Running:**
   ```bash
   cd "G:\Projects\Centralize Data Centre"
   .\check_services_status.bat
   ```

2. **Restart Broadcaster:**
   ```bash
   cd "G:\Projects\Centralize Data Centre"
   .\start_all_services.bat
   ```

3. **Check Port:**
   ```powershell
   netstat -an | findstr ":8765"
   ```
   Should show: `LISTENING` status

4. **Test Connection:**
   ```bash
   py test_ws_connection.py
   ```

## Next Steps
Once broadcaster is confirmed running, the WebSocket connection should work automatically.

