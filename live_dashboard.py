"""
NiceGUI Live Paper Trading Dashboard

Starts a local dashboard (default http://localhost:8080) and provides buttons to:
- Connect to broadcaster + start paper trading
- Stop (disconnect + stop background loop)
- Export completed trades to CSV/XLSX (timestamped files, safe with Excel open)

Shows:
- Feed timestamp (from broadcaster ts) in IST
- Connection status (connected, last msg age, reconnect count)
- Recent logs (signals/patterns/entries/exits)
- Active trade (entry, current price, TG/SL, running P&L)
- Trades table + summary totals
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from nicegui import ui

from broadcaster_client import BroadcasterClientConfig, BroadcasterWebSocketClient
from live_trading_engine import EngineConfig, LiveTradingEngine, PaperTrade, to_ist_str, parse_ts_utc, safe_float, fill_price
from trade_store import TradeStore

import subprocess
import sqlite3


# Project root directory
PROJECT_ROOT = r"g:\Projects\NIFTY_Options_Backtest"


class ProcessManager:
    """Manages subprocess for Live mode data collector and paper trading engine."""
    
    def __init__(self):
        self.data_collector_process: Optional[subprocess.Popen] = None
        self.trading_process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
    
    def _is_process_running_by_name(self, script_name: str) -> bool:
        """Check if a process is running by script name (works for any process, not just dashboard-started)."""
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline')
                    if cmdline and any(script_name in str(cmd) for cmd in cmdline):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            # Fallback if psutil not available
            pass
        return False
    
    def start_data_collector(self) -> bool:
        """Start the data collector subprocess using batch file."""
        with self.lock:
            # Check if already running (any instance)
            if self._is_process_running_by_name("nifty_stream_local_sqlite.py"):
                return False  # Already running
            
            try:
                # Use the batch file which keeps console open
                batch_file = os.path.join(PROJECT_ROOT, "scripts", "start_live_data_collector.bat")
                self.data_collector_process = subprocess.Popen(
                    [batch_file],
                    cwd=PROJECT_ROOT,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                return True
            except Exception as e:
                print(f"Error starting data collector: {e}")
                return False
    
    def stop_data_collector(self) -> bool:
        """Stop the data collector subprocess."""
        with self.lock:
            stopped = False
            # Try to stop dashboard-started process
            if self.data_collector_process is not None:
                try:
                    self.data_collector_process.terminate()
                    self.data_collector_process.wait(timeout=5)
                    stopped = True
                except:
                    try:
                        self.data_collector_process.kill()
                        stopped = True
                    except:
                        pass
                self.data_collector_process = None
            
            # Also try to kill any running instance by name
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline')
                        if cmdline and any("nifty_stream_local_sqlite.py" in str(cmd) for cmd in cmdline):
                            proc.terminate()
                            stopped = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                pass
            
            return stopped
    
    def start_trading(self, db_path: str) -> bool:
        """Start the paper trading engine subprocess using batch file."""
        with self.lock:
            # Check if already running (any instance)
            if self._is_process_running_by_name("paper_trading_engine.py"):
                return False  # Already running
            
            try:
                # Use the batch file which keeps console open
                batch_file = os.path.join(PROJECT_ROOT, "scripts", "start_live_paper_trading.bat")
                self.trading_process = subprocess.Popen(
                    [batch_file],
                    cwd=PROJECT_ROOT,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                return True
            except Exception as e:
                print(f"Error starting trading engine: {e}")
                return False
    
    def stop_trading(self) -> bool:
        """Stop the paper trading engine subprocess."""
        with self.lock:
            stopped = False
            # Try to stop dashboard-started process
            if self.trading_process is not None:
                try:
                    self.trading_process.terminate()
                    self.trading_process.wait(timeout=5)
                    stopped = True
                except:
                    try:
                        self.trading_process.kill()
                        stopped = True
                    except:
                        pass
                self.trading_process = None
            
            # Also try to kill any running instance by name
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline')
                        if cmdline and any("paper_trading_engine.py" in str(cmd) for cmd in cmdline):
                            proc.terminate()
                            stopped = True
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                pass
            
            return stopped
    
    def is_data_collector_running(self) -> bool:
        """Check if data collector is running (any instance)."""
        with self.lock:
            # First check dashboard-started process
            if self.data_collector_process is not None and self.data_collector_process.poll() is None:
                return True
            # Then check for any running instance
            return self._is_process_running_by_name("nifty_stream_local_sqlite.py")
    
    def is_trading_running(self) -> bool:
        """Check if trading engine is running (any instance)."""
        with self.lock:
            # First check dashboard-started process
            if self.trading_process is not None and self.trading_process.poll() is None:
                return True
            # Then check for any running instance
            return self._is_process_running_by_name("paper_trading_engine.py")


# Global process manager
process_manager = ProcessManager()


def get_database_stats(db_path: str) -> dict:
    """Get statistics from a database."""
    try:
        if not os.path.exists(db_path):
            return {"exists": False, "records": 0, "last_update": None}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), MAX(ts) FROM ltp_ticks")
        row = cursor.fetchone()
        count = row[0] or 0
        last_ts = row[1] if row[1] else None
        conn.close()
        
        # Parse last timestamp to IST
        last_update_ist = None
        if last_ts:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
                last_update_ist = dt.astimezone(IST).strftime("%H:%M:%S")
            except:
                last_update_ist = last_ts[-8:] if len(last_ts) >= 8 else last_ts
        
        return {"exists": True, "records": count, "last_update": last_update_ist}
    except Exception as e:
        return {"exists": False, "records": 0, "last_update": None, "error": str(e)}


class AppState:
    def __init__(self) -> None:
        self.lock = threading.RLock()

        self.client: Optional[BroadcasterWebSocketClient] = None
        self.engine: Optional[LiveTradingEngine] = None
        self.store: Optional[TradeStore] = None

        self.running = False
        self.stop_evt = threading.Event()
        self.worker: Optional[threading.Thread] = None

        self.msg_q: "queue.Queue[Any]" = queue.Queue(maxsize=50000)
        self.reconnect_events = 0
        self._last_conn_state: Optional[bool] = None

        self.logs: List[str] = []
        self.max_logs = 300

        self.last_export_path: str = ""
        
        # Mode selection: "live" or "replay"
        self.mode: str = "live"
        
        # Track last log file read time (for throttling)
        self.last_log_read_time: float = 0.0

    def push_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs :]


state = AppState()

DEFAULTS = {
    # Broadcaster
    "ws_url": "ws://127.0.0.1:8765",
    # Strategy (match current backtest defaults that produced good results)
    "lot_size": 3750,
    "target_pct": 10.0,
    "stop_pct": 5.0,
    "max_hold_minutes": 3.0,
    # Execution realism defaults (baseline backtest settings)
    "use_bid_ask": False,
    "slippage_points": 0.0,
    "latency_seconds": 0.0,
    # Signal/pattern settings (fixed pipeline)
    "candle_interval_seconds": 30,
    "movement_threshold": 0.11,
    "pattern_window_seconds": 60,
}


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

IST = timezone(timedelta(hours=5, minutes=30))


def export_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(d, exist_ok=True)
    return d


def db_path() -> str:
    """Returns trade database path (for storing completed trades)."""
    d = export_dir()
    return os.path.join(d, "live_trades.db")


def market_db_path() -> str:
    """Returns market data database path based on current mode."""
    if state.mode == "live":
        return r"g:\Projects\NIFTY_Options_Backtest\data\nifty_live.db"
    else:  # replay
        return r"g:\Projects\NIFTY_Options_Backtest\data\nifty_replay.db"


def default_market_db_path() -> str:
    """Legacy function - now uses market_db_path()."""
    return market_db_path()


def on_ws_message(msg: Dict[str, Any]) -> None:
    try:
        state.msg_q.put_nowait(msg)
    except Exception:
        return


def start_worker(ws_url: str, cfg: EngineConfig) -> None:
    with state.lock:
        if state.running:
            return

        state.stop_evt.clear()
        state.running = True
        state.reconnect_events = 0
        state._last_conn_state = None

        # Create engine with UI log callback
        state.store = TradeStore(db_path())

        state.engine = LiveTradingEngine(
            cfg,
            log_fn=state.push_log,
            trade_store=state.store,
        )

        # Create WS client
        state.client = BroadcasterWebSocketClient(
            config=BroadcasterClientConfig(ws_url=ws_url),
            on_data=on_ws_message,
        )
        state.client.connect()

        def loop() -> None:
            while not state.stop_evt.is_set():
                client = state.client
                eng = state.engine
                if not client or not eng:
                    time.sleep(0.2)
                    continue

                # track reconnect-ish events by connection state changes
                conn = client.is_connected()
                if state._last_conn_state is None:
                    state._last_conn_state = conn
                elif conn != state._last_conn_state:
                    state.reconnect_events += 1
                    state._last_conn_state = conn

                try:
                    raw = state.msg_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                payloads: List[Dict[str, Any]] = []
                if isinstance(raw, list):
                    payloads = [x for x in raw if isinstance(x, dict)]
                elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
                    payloads = [x for x in raw.get("data") if isinstance(x, dict)]
                elif isinstance(raw, dict):
                    payloads = [raw]

                syms = client.get_symbols()
                for tick in payloads:
                    eng.on_tick(tick, syms)

            # cleanup
            try:
                if state.client:
                    state.client.close()
            finally:
                with state.lock:
                    if state.store:
                        try:
                            state.store.close()
                        except Exception:
                            pass
                        state.store = None
                    state.client = None
                    state.engine = None
                    state.running = False

        state.worker = threading.Thread(target=loop, name="dashboard-worker", daemon=True)
        state.worker.start()


def stop_worker() -> None:
    with state.lock:
        if not state.running:
            return
        state.stop_evt.set()


def compute_active_pnl_snapshot() -> Dict[str, Any]:
    with state.lock:
        eng = state.engine
        if not eng:
            return {"status": "STOPPED"}
        snap = eng.snapshot()
        active: Optional[PaperTrade] = snap["active_trade"]
        pending: Optional[PaperTrade] = snap["pending_trade"]

        client = state.client
        symbols = client.get_symbols() if client else []
        connected = client.is_connected() if client else False
        last_age = client.last_message_age_s() if client else None

        out: Dict[str, Any] = {
            "connected": connected,
            "last_msg_age": last_age,
            "reconnect_events": state.reconnect_events,
            "feed_ts_ist": snap.get("last_feed_ts_ist") or "",
            "spot": snap.get("last_spot"),
            "symbols": len(symbols),
            "warmup": snap.get("warmup") or {},
            "feed_ts_utc": snap.get("last_feed_ts_utc"),
        }

        if pending and not active:
            out["trade_state"] = "PENDING"
            out["trade"] = pending
            return out

        if not active:
            out["trade_state"] = "NONE"
            return out

        # Compute running PnL using latest tick
        out["trade_state"] = "ACTIVE"
        out["trade"] = active

        latest_tick = None
        if client:
            latest_tick = client.get_latest(active.option_symbol)

        if latest_tick and active.entry_price:
            cur_exit = fill_price(latest_tick, "SELL", eng.cfg.use_bid_ask, eng.cfg.slippage_points)
            if cur_exit is None:
                cur_exit = safe_float(latest_tick.get("ltp")) or 0.0
            entry = float(active.entry_price)
            points = float(cur_exit) - entry
            gross = points * float(eng.cfg.lot_size)
            buy_val = entry * float(eng.cfg.lot_size)
            sell_val = float(cur_exit) * float(eng.cfg.lot_size)
            # cost estimate uses same function inside engine, but we avoid extra import loops
            from live_trading_engine import calculate_transaction_cost

            tc = calculate_transaction_cost(buy_val, sell_val)
            net = gross - tc
            target = entry * (1.0 + eng.cfg.target_pct / 100.0)
            stop = entry * (1.0 - eng.cfg.stop_pct / 100.0)
            out.update(
                {
                    "cur": float(cur_exit),
                    "run_points": points,
                    "run_net": net,
                    "tg": target,
                    "sl": stop,
                }
            )
        return out


def feed_day_start_utc(feed_ts_utc: Optional[datetime]) -> Optional[datetime]:
    """Compute 00:00 IST of the feed-day, converted to UTC."""
    if not feed_ts_utc:
        return None
    if feed_ts_utc.tzinfo is None:
        feed_ts_utc = feed_ts_utc.replace(tzinfo=timezone.utc)
    ist_dt = feed_ts_utc.astimezone(IST)
    ist_start = ist_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return ist_start.astimezone(timezone.utc)


def ist_range_to_utc_iso(start_date: str, start_time: str, end_date: str, end_time: str) -> Optional[Dict[str, str]]:
    """
    Convert IST date/time strings to UTC ISO strings suitable for sqlite TEXT range queries.
    Expected formats:
      date: YYYY-MM-DD
      time: HH:MM or HH:MM:SS
    Returns {start_utc, end_utc} or None if parse fails.
    """
    def parse_dt(d: str, t: str) -> Optional[datetime]:
        t = t.strip()
        if len(t.split(":")) == 2:
            t = t + ":00"
        try:
            dt = datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=IST)
        except Exception:
            return None

    s = parse_dt(start_date, start_time)
    e = parse_dt(end_date, end_time)
    if not s or not e:
        return None
    s_utc = s.astimezone(timezone.utc)
    e_utc = e.astimezone(timezone.utc)
    return {
        "start_utc": s_utc.isoformat(timespec="seconds"),
        "end_utc": e_utc.isoformat(timespec="seconds"),
    }


def fmt_money(x: Any) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "-"


def fmt_num(x: Any, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def positions_open_rows(snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One-row open position table (since engine enforces single active trade)."""
    trade_state = snap.get("trade_state")
    if trade_state != "ACTIVE":
        return []
    t: PaperTrade = snap["trade"]
    entry = float(t.entry_price or 0)
    cur = float(snap.get("cur") or entry)
    qty = int(getattr(state.engine.cfg, "lot_size", 0)) if state.engine else 0  # type: ignore
    pnl = float(snap.get("run_net") or 0)
    chg_pct = ((cur - entry) / entry) * 100.0 if entry > 0 else 0.0
    return [
        {
            "Product": "NRML",
            "Instrument": t.option_symbol,
            "Qty": qty,
            "Avg": fmt_num(entry, 2),
            "LTP": fmt_num(cur, 2),
            "P&L": fmt_money(pnl),
            "Chg%": fmt_num(chg_pct, 2),
        }
    ]


def closed_today_rows(feed_ts_utc: Optional[datetime], limit: int = 500) -> List[Dict[str, Any]]:
    with state.lock:
        store = state.store
    if not store:
        return []
    day_start = feed_day_start_utc(feed_ts_utc)
    since = day_start.isoformat() if day_start else None
    rows = store.list_trades(limit=limit, since_utc=since)
    out = []
    for r in rows:
        try:
            entry = float(r.get("entry_price") or 0)
            exitp = float(r.get("exit_price") or 0)
            chg_pct = ((exitp - entry) / entry) * 100.0 if entry > 0 else 0.0
        except Exception:
            chg_pct = 0.0
        out.append(
            {
                "Order ID": r.get("order_id"),
                "Instrument": r.get("option_symbol"),
                "Entry (IST)": to_ist_str(parse_ts_utc(r.get("entry_time_utc"))),
                "Entry Price": fmt_num(r.get("entry_price"), 2),
                "Exit (IST)": to_ist_str(parse_ts_utc(r.get("exit_time_utc"))),
                "Exit Price": fmt_num(r.get("exit_price"), 2),
                "Net P&L": fmt_money(r.get("net_pnl")),
                "Chg%": fmt_num(chg_pct, 2),
                "Reason": r.get("exit_reason") or "",
            }
        )
    return out


def closed_today_total_net(rows: List[Dict[str, Any]]) -> float:
    total = 0.0
    for r in rows:
        try:
            total += float(str(r.get("Net P&L")).replace(",", ""))
        except Exception:
            pass
    return total


def trades_table_rows(trades: List[PaperTrade]) -> List[Dict[str, Any]]:
    rows = []
    for t in trades:
        rows.append(
            {
                "Order ID": t.order_id,
                "Entry (IST)": to_ist_str(t.entry_time_utc),
                "Exit (IST)": to_ist_str(t.exit_time_utc),
                "Symbol": t.option_symbol,
                "Dir": t.predicted_direction,
                "Entry": round(float(t.entry_price or 0), 2),
                "Exit": round(float(t.exit_price or 0), 2),
                "Pts": round(float(t.pnl_points or 0), 2),
                "Net": round(float(t.net_pnl or 0), 2),
                "Reason": t.exit_reason or "",
                "Patterns": t.patterns_str,
            }
        )
    return rows


def summary_totals(trades: List[PaperTrade]) -> Dict[str, Any]:
    net = sum(float(t.net_pnl or 0) for t in trades)
    wins = sum(1 for t in trades if float(t.net_pnl or 0) > 0)
    losses = sum(1 for t in trades if float(t.net_pnl or 0) < 0)
    return {"trades": len(trades), "wins": wins, "losses": losses, "net": net}


# ---------------- UI ----------------

ui.page_title("NIFTY Live Paper Trading Dashboard")
ui.dark_mode().enable()

with ui.header().classes("items-center justify-between"):
    ui.label("NIFTY Live Paper Trading Dashboard").classes("text-lg")


# Mode Selection Card - Unified Interface
with ui.row().classes("w-full gap-4"):
    # Live Mode Card
    with ui.card().classes("w-1/2 cursor-pointer") as live_card:
        with ui.row().classes("items-center gap-2"):
            ui.icon("analytics", size="lg").classes("text-blue-500")
            ui.label("LIVE MODE").classes("text-h6 font-bold text-blue-500")
        ui.label("Real-time Angel One API data collection").classes("text-sm")
        ui.label("Database: nifty_live.db").classes("text-caption text-grey")
        live_status_badge = ui.badge("SELECTED", color="blue").classes("mt-2")
        
    # Replay Mode Card
    with ui.card().classes("w-1/2 cursor-pointer") as replay_card:
        with ui.row().classes("items-center gap-2"):
            ui.icon("replay", size="lg").classes("text-teal-500")
            ui.label("REPLAY MODE").classes("text-h6 font-bold text-teal-500")
        ui.label("Historical data from broadcaster").classes("text-sm")
        ui.label("Database: nifty_replay.db").classes("text-caption text-grey")
        replay_status_badge = ui.badge("", color="grey").classes("mt-2")

# Initialize mode state
state.mode = "live"

# Mode selection handlers
def select_live_mode():
    state.mode = "live"
    state.push_log("UI: Switched to LIVE mode")
    live_status_badge.set_text("SELECTED")
    live_status_badge.props("color=blue")
    replay_status_badge.set_text("")
    replay_status_badge.props("color=grey")
    live_card.classes(replace="w-1/2 cursor-pointer border-2 border-blue-500")
    replay_card.classes(replace="w-1/2 cursor-pointer")
    update_dynamic_controls()

def select_replay_mode():
    state.mode = "replay"
    state.push_log("UI: Switched to REPLAY mode")
    replay_status_badge.set_text("SELECTED")
    replay_status_badge.props("color=teal")
    live_status_badge.set_text("")
    live_status_badge.props("color=grey")
    replay_card.classes(replace="w-1/2 cursor-pointer border-2 border-teal-500")
    live_card.classes(replace="w-1/2 cursor-pointer")
    update_dynamic_controls()

live_card.on("click", lambda: select_live_mode())
replay_card.on("click", lambda: select_replay_mode())

# Database Status Display
with ui.row().classes("w-full"):
    with ui.card().classes("w-full"):
        with ui.row().classes("items-center justify-between"):
            ui.label("📊 Database Status").classes("font-bold")
            db_refresh_btn = ui.button("Refresh", icon="refresh", on_click=lambda: update_db_status()).props("flat dense")
        
        with ui.row().classes("gap-8"):
            with ui.column():
                ui.label("Live Database").classes("text-caption")
                live_db_status = ui.label("Records: - | Last: -").classes("text-sm")
            with ui.column():
                ui.label("Replay Database").classes("text-caption")
                replay_db_status = ui.label("Records: - | Last: -").classes("text-sm")

def update_db_status():
    # Live DB stats
    live_stats = get_database_stats(r"g:\Projects\NIFTY_Options_Backtest\data\nifty_live.db")
    if live_stats["exists"]:
        live_db_status.set_text(f"Records: {live_stats['records']:,} | Last: {live_stats['last_update'] or '-'}")
    else:
        live_db_status.set_text("Database not found")
    
    # Replay DB stats
    replay_stats = get_database_stats(r"g:\Projects\NIFTY_Options_Backtest\data\nifty_replay.db")
    if replay_stats["exists"]:
        replay_db_status.set_text(f"Records: {replay_stats['records']:,} | Last: {replay_stats['last_update'] or '-'}")
    else:
        replay_db_status.set_text("Database not found")

# Dynamic Controls Container
controls_container = ui.card().classes("w-full")

def update_dynamic_controls():
    controls_container.clear()
    with controls_container:
        if state.mode == "live":
            render_live_controls()
        else:
            render_replay_controls()

def render_live_controls():
    """Render controls specific to Live mode."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("analytics", color="blue")
        ui.label("LIVE MODE CONTROLS").classes("text-h6 font-bold")
    
    # Data Collector Section
    with ui.expansion("📡 Data Collector", icon="sensors", value=True).classes("w-full"):
        data_collector_status = ui.label("Status: Checking...")
        with ui.row():
            start_dc_btn = ui.button("Start Data Collector", icon="play_arrow", color="primary")
            stop_dc_btn = ui.button("Stop Data Collector", icon="stop", color="negative")
        
        def update_dc_button_states():
            """Update button states based on process status."""
            is_running = process_manager.is_data_collector_running()
            start_dc_btn.set_enabled(not is_running)
            stop_dc_btn.set_enabled(is_running)
            if is_running:
                data_collector_status.set_text("Status: 🟢 Running")
            else:
                data_collector_status.set_text("Status: 🔴 Not running")
        
        def start_data_collector():
            if process_manager.start_data_collector():
                data_collector_status.set_text("Status: 🟢 Started (check new console window)")
                state.push_log("UI: Data collector started")
                ui.notify("Data collector started", type="positive")
                update_dc_button_states()
            else:
                ui.notify("Data collector already running or failed to start", type="warning")
        
        def stop_data_collector():
            if process_manager.stop_data_collector():
                data_collector_status.set_text("Status: 🔴 Stopped")
                state.push_log("UI: Data collector stopped")
                ui.notify("Data collector stopped", type="info")
                update_dc_button_states()
            else:
                ui.notify("Data collector not running", type="warning")
        
        start_dc_btn.on("click", lambda: start_data_collector())
        stop_dc_btn.on("click", lambda: stop_data_collector())
        
        # Initial status check
        update_dc_button_states()
        
        # Periodic status update (every 3 seconds)
        ui.timer(3.0, lambda: update_dc_button_states())
    
    # Paper Trading Engine Section
    with ui.expansion("🤖 Paper Trading Engine", icon="smart_toy", value=True).classes("w-full"):
        trading_status = ui.label("Status: Checking...")
        with ui.row():
            start_trading_btn = ui.button("Start Trading", icon="play_arrow", color="primary")
            stop_trading_btn = ui.button("Stop Trading", icon="stop", color="negative")
        
        def update_trading_button_states():
            """Update button states based on process status."""
            is_running = process_manager.is_trading_running()
            start_trading_btn.set_enabled(not is_running)
            stop_trading_btn.set_enabled(is_running)
            if is_running:
                trading_status.set_text("Status: 🟢 Running")
            else:
                trading_status.set_text("Status: 🔴 Not running")
        
        def start_trading():
            db = r"g:\Projects\NIFTY_Options_Backtest\data\nifty_live.db"
            if process_manager.start_trading(db):
                trading_status.set_text("Status: 🟢 Started (check new console window)")
                state.push_log("UI: Paper trading engine started")
                ui.notify("Paper trading engine started", type="positive")
                update_trading_button_states()
            else:
                ui.notify("Trading engine already running or failed to start", type="warning")
        
        def stop_trading():
            if process_manager.stop_trading():
                trading_status.set_text("Status: 🔴 Stopped")
                state.push_log("UI: Paper trading engine stopped")
                ui.notify("Paper trading engine stopped", type="info")
                update_trading_button_states()
            else:
                ui.notify("Trading engine not running", type="warning")
        
        start_trading_btn.on("click", lambda: start_trading())
        stop_trading_btn.on("click", lambda: stop_trading())
        
        # Initial status check
        update_trading_button_states()
        
        # Periodic status update (every 3 seconds)
        ui.timer(3.0, lambda: update_trading_button_states())
    
    # Strategy Settings
    render_strategy_settings()

def render_replay_controls():
    """Render controls specific to Replay mode."""
    with ui.row().classes("items-center gap-2"):
        ui.icon("replay", color="teal")
        ui.label("REPLAY MODE CONTROLS").classes("text-h6 font-bold")
    
    # Broadcaster Connection Section
    with ui.expansion("📡 Broadcaster Connection", icon="cell_tower", value=True).classes("w-full"):
        global ws_url_in
        ws_url_in = ui.input("WebSocket URL", value=DEFAULTS["ws_url"]).classes("w-full")
        
        replay_conn_status = ui.label("Status: Disconnected")
        
        with ui.row():
            global start_btn, stop_btn
            start_btn = ui.button("Connect + Start Trading", icon="play_arrow", color="primary")
            stop_btn = ui.button("Stop", icon="stop", color="negative")
        
        def update_replay_button_states():
            """Update button states based on connection status."""
            is_connected = state.running
            start_btn.set_enabled(not is_connected)
            stop_btn.set_enabled(is_connected)
            if is_connected:
                replay_conn_status.set_text("Status: 🟢 Connected")
            else:
                replay_conn_status.set_text("Status: 🔴 Disconnected")
        
        def on_replay_start():
            try:
                cfg = read_cfg_from_inputs()
                start_worker(ws_url=str(ws_url_in.value), cfg=cfg)
                replay_conn_status.set_text("Status: 🟢 Connected")
                state.push_log(f"UI: Started Replay mode with ws_url={ws_url_in.value}")
                ui.notify("Connected to broadcaster", type="positive")
                update_replay_button_states()
            except Exception as e:
                replay_conn_status.set_text(f"Status: 🔴 Error - {e}")
                ui.notify(f"Connection failed: {e}", type="negative")
        
        def on_replay_stop():
            stop_worker()
            replay_conn_status.set_text("Status: 🔴 Disconnected")
            state.push_log("UI: Stopped Replay mode")
            ui.notify("Disconnected from broadcaster", type="info")
            update_replay_button_states()
        
        start_btn.on("click", lambda: on_replay_start())
        stop_btn.on("click", lambda: on_replay_stop())
        
        # Initial status check
        update_replay_button_states()
        
        # Periodic status update (every 2 seconds)
        ui.timer(2.0, lambda: update_replay_button_states())
    
    # Strategy Settings
    render_strategy_settings()

def render_strategy_settings():
    """Render common strategy settings."""
    with ui.expansion("⚙️ Strategy Settings", icon="settings", value=False).classes("w-full"):
        with ui.row():
            global lot_in, tg_in, sl_in, hold_in
            lot_in = ui.number("Lot size", value=DEFAULTS["lot_size"], format="%.0f")
            tg_in = ui.number("Target %", value=DEFAULTS["target_pct"], format="%.2f")
            sl_in = ui.number("Stop %", value=DEFAULTS["stop_pct"], format="%.2f")
            hold_in = ui.number("Max hold (min)", value=DEFAULTS["max_hold_minutes"], format="%.2f")

        with ui.row():
            global use_ba_in, slip_in, lat_in
            use_ba_in = ui.checkbox("Use bid/ask fills", value=DEFAULTS["use_bid_ask"])
            slip_in = ui.number("Slippage (pts)", value=DEFAULTS["slippage_points"], format="%.2f")
            lat_in = ui.number("Latency (s)", value=DEFAULTS["latency_seconds"], format="%.2f")

        with ui.row():
            global candle_in, move_in, win_in
            candle_in = ui.number("Candle interval (s)", value=DEFAULTS["candle_interval_seconds"], format="%.0f")
            move_in = ui.number("Movement threshold (%)", value=DEFAULTS["movement_threshold"], format="%.3f")
            win_in = ui.number("Pattern window (s)", value=DEFAULTS["pattern_window_seconds"], format="%.0f")

        with ui.row():
            global reset_btn, export_csv_btn, export_xlsx_btn
            reset_btn = ui.button("Reset to Defaults", icon="restart_alt", color="accent")
            export_csv_btn = ui.button("Export CSV", icon="download", color="secondary")
            export_xlsx_btn = ui.button("Export Excel", icon="table_chart", color="secondary")
        
        global export_lbl
        export_lbl = ui.label("")
        
        # Wire up button handlers
        reset_btn.on("click", lambda: reset_to_defaults())
        export_csv_btn.on("click", lambda: on_export_csv())
        export_xlsx_btn.on("click", lambda: on_export_xlsx())

# Initialize controls
update_dynamic_controls()
update_db_status()

# Set initial visual state for Live mode
live_card.classes(add="border-2 border-blue-500")




# Note: Controls are now dynamically rendered above in the controls_container
# based on the selected mode (Live or Replay)


tabs = ui.tabs().classes("w-full")
with tabs:
    tab_active = ui.tab("Active")
    tab_history = ui.tab("History (DB)")
    tab_logs = ui.tab("Logs")

with ui.tab_panels(tabs, value=tab_active).classes("w-full"):
    with ui.tab_panel(tab_active):
        with ui.row().classes("w-full"):
            with ui.card().classes("w-1/2"):
                ui.label("Connection / Feed")
                conn_lbl = ui.label("Status: STOPPED")
                feed_lbl = ui.label("Feed time (IST): -")
                spot_lbl = ui.label("Spot: -")
                sym_lbl = ui.label("Symbols: -")

            with ui.card().classes("w-1/2"):
                ui.label("Trade Controls")
                trade_lbl = ui.label("No active trade")
                with ui.row():
                    cancel_btn = ui.button("Cancel Pending", color="warning")
                    exit_btn = ui.button("Manual Exit Active", color="negative")

        with ui.row().classes("w-full"):
            with ui.card().classes("w-1/2"):
                ui.label("Open Positions (feed-day)")
                open_positions_table = ui.table(
                    columns=[
                        {"name": "Product", "label": "Product", "field": "Product"},
                        {"name": "Instrument", "label": "Instrument", "field": "Instrument"},
                        {"name": "Qty", "label": "Qty", "field": "Qty", "sortable": True},
                        {"name": "Avg", "label": "Avg", "field": "Avg", "sortable": True},
                        {"name": "LTP", "label": "LTP", "field": "LTP", "sortable": True},
                        {"name": "P&L", "label": "P&L", "field": "P&L", "sortable": True},
                        {"name": "Chg%", "label": "Chg%", "field": "Chg%", "sortable": True},
                    ],
                    rows=[],
                    row_key="Instrument",
                ).classes("w-full")
                # Colorize P&L / Chg% like broker UI
                open_positions_table.add_slot(
                    "body-cell-P&L",
                    r"""
<q-td :props="props">
  <span :style="{color: (parseFloat(String(props.value).replace(/,/g,'')) >= 0 ? '#2ecc71' : '#ff4d4f'), fontWeight: '600'}">
    {{ props.value }}
  </span>
</q-td>
""",
                )
                open_positions_table.add_slot(
                    "body-cell-Chg%",
                    r"""
<q-td :props="props">
  <span :style="{color: (parseFloat(props.value) >= 0 ? '#2ecc71' : '#ff4d4f'), fontWeight: '600'}">
    {{ props.value }}%
  </span>
</q-td>
""",
                )
                open_total_lbl = ui.label("Open P&L: 0.00")

            with ui.card().classes("w-1/2"):
                ui.label("Closed Trades (feed-day)")
                closed_table = ui.table(
                    columns=[
                        {"name": "Order ID", "label": "Order ID", "field": "Order ID", "sortable": True},
                        {"name": "Instrument", "label": "Instrument", "field": "Instrument", "sortable": True},
                        {"name": "Entry (IST)", "label": "Entry (IST)", "field": "Entry (IST)", "sortable": True},
                        {"name": "Entry Price", "label": "Entry Price", "field": "Entry Price", "sortable": True},
                        {"name": "Exit (IST)", "label": "Exit (IST)", "field": "Exit (IST)", "sortable": True},
                        {"name": "Exit Price", "label": "Exit Price", "field": "Exit Price", "sortable": True},
                        {"name": "Net P&L", "label": "Net P&L", "field": "Net P&L", "sortable": True},
                        {"name": "Chg%", "label": "Chg%", "field": "Chg%", "sortable": True},
                        {"name": "Reason", "label": "Reason", "field": "Reason", "sortable": True},
                    ],
                    rows=[],
                    row_key="Order ID",
                ).classes("w-full")
                closed_table.add_slot(
                    "body-cell-Net P&L",
                    r"""
<q-td :props="props">
  <span :style="{color: (parseFloat(String(props.value).replace(/,/g,'')) >= 0 ? '#2ecc71' : '#ff4d4f'), fontWeight: '600'}">
    {{ props.value }}
  </span>
</q-td>
""",
                )
                closed_table.add_slot(
                    "body-cell-Chg%",
                    r"""
<q-td :props="props">
  <span :style="{color: (parseFloat(props.value) >= 0 ? '#2ecc71' : '#ff4d4f'), fontWeight: '600'}">
    {{ props.value }}%
  </span>
</q-td>
""",
                )
                closed_total_lbl = ui.label("Closed P&L: 0.00")

    with ui.tab_panel(tab_history):
        with ui.card().classes("w-full"):
            ui.label(f"Trade History from SQLite: {db_path()}")
            with ui.row():
                hist_limit = ui.number("Rows", value=500, format="%.0f")
                hist_refresh = ui.button("Refresh", color="primary")
            with ui.row():
                ui.label("Filter/Delete by Exit Time (IST):")
            with ui.row():
                hist_start_date = ui.input("Start date (YYYY-MM-DD)", value="")
                hist_start_time = ui.input("Start time (HH:MM)", value="09:15")
                hist_end_date = ui.input("End date (YYYY-MM-DD)", value="")
                hist_end_time = ui.input("End time (HH:MM)", value="15:30")
            with ui.row():
                hist_apply_range = ui.button("Apply Range Filter", color="secondary")
                hist_clear_range = ui.button("Clear Filter", color="secondary")
                hist_delete_range = ui.button("Delete Trades in Range", color="negative")
            hist_table = ui.table(columns=[], rows=[]).classes("w-full")

    with ui.tab_panel(tab_logs):
        # Sub-tabs for different log types
        log_subtabs = ui.tabs().classes("w-full")
        with log_subtabs:
            log_tab_all = ui.tab("All Logs")
            log_tab_patterns = ui.tab("Pattern Analysis")
            log_tab_signals = ui.tab("Signal Generation")
        
        with ui.tab_panels(log_subtabs, value=log_tab_all).classes("w-full"):
            # All Logs
            with ui.tab_panel(log_tab_all):
                with ui.card().classes("w-full"):
                    ui.label("Live Log (feed timestamps inside messages)").classes("font-bold")
                    
                    # Mode-aware message
                    global log_mode_message
                    log_mode_message = ui.label("")
                    
                    def update_log_mode_message():
                        if state.mode == "live":
                            log_mode_message.set_text("ℹ️ LIVE MODE: Logs appear in the console windows (Data Collector & Paper Trading Engine). Check those windows for real-time activity.")
                            log_mode_message.classes("text-orange-500 font-bold")
                        else:
                            log_mode_message.set_text("📊 REPLAY MODE: Logs appear below as trading activity occurs.")
                            log_mode_message.classes("text-teal-500")
                    
                    update_log_mode_message()
                    ui.timer(2.0, lambda: update_log_mode_message())
                    
                    log_box = ui.log(max_lines=300).classes("w-full").style("height: 600px")
            
            # Pattern Analysis Logs
            with ui.tab_panel(log_tab_patterns):
                with ui.card().classes("w-full"):
                    ui.label("Pattern Analysis Details").classes("font-bold")
                    ui.label("Shows candle analysis, direction calculations, and threshold tracking").classes("text-caption")
                    pattern_log_box = ui.log(max_lines=200).classes("w-full").style("height: 600px")
                    
                    # Add explanation card
                    with ui.expansion("📖 Pattern Analysis Guide", icon="help").classes("w-full mt-4"):
                        ui.markdown("""
### Pattern Analysis Explained

Shows the ongoing analysis of each candle:

**Candle Details:**
- Open, High, Low, Close prices
- Percentage change calculation
- Direction determination (UP/DOWN/NEUTRAL)

**Trend Tracking:**
- Start price for cumulative tracking
- Current cumulative move percentage
- Threshold comparison (0.11%)
- Whether threshold has been crossed

This tab shows ALL candle processing, even when no signal is generated.
                        """)
            
            # Signal Generation Logs
            with ui.tab_panel(log_tab_signals):
                with ui.card().classes("w-full"):
                    ui.label("Signal Generation Logic").classes("font-bold")
                    ui.label("Shows signals and pattern detection (appears only when threshold is crossed)").classes("text-caption")
                    signal_log_box = ui.log(max_lines=200).classes("w-full").style("height: 600px")
                    
                    # Add explanation card
                    with ui.expansion("📖 Signal Generation Guide", icon="help").classes("w-full mt-4"):
                        ui.markdown("""
### Signal Generation Explained

**Trend Detection (30-second candles):**
1. Calculate cumulative price movement
2. Check if movement exceeds 0.11% threshold
3. Determine direction (UP/DOWN)

**Signal Flow:**
```
Market Data → Candle Building → Trend Detection → Pattern Analysis → Trade Signal
```

**Why No Signal?**
- ❌ **No Trend**: Movement < 0.11%
- ❌ **Insufficient Data**: Less than 60 seconds of data
- ❌ **Pattern Failed**: Less than 3 indicators agree
- ❌ **Already in Trade**: One trade at a time

**Log Format:**
- 🔍 **Trend Detected** - Shows direction and magnitude
- ⏳ **Waiting for Pattern** - Trend found, analyzing pattern
- ⛔ **No Trend** - Movement below threshold
                        """)



def read_cfg_from_inputs() -> EngineConfig:
    return EngineConfig(
        lot_size=int(lot_in.value or 3750),
        target_pct=float(tg_in.value or 10.0),
        stop_pct=float(sl_in.value or 5.0),
        max_hold_minutes=float(hold_in.value or 3.0),
        use_bid_ask=bool(use_ba_in.value),
        slippage_points=float(slip_in.value or 0.0),
        latency_seconds=float(lat_in.value or 0.0),
        candle_interval_seconds=int(candle_in.value or 30),
        movement_threshold=float(move_in.value or 0.11),
        pattern_window_seconds=int(win_in.value or 60),
    )

def reset_to_defaults() -> None:
    # Only reset the form; does not stop/start the engine.
    ws_url_in.value = DEFAULTS["ws_url"]
    lot_in.value = DEFAULTS["lot_size"]
    tg_in.value = DEFAULTS["target_pct"]
    sl_in.value = DEFAULTS["stop_pct"]
    hold_in.value = DEFAULTS["max_hold_minutes"]
    use_ba_in.value = DEFAULTS["use_bid_ask"]
    slip_in.value = DEFAULTS["slippage_points"]
    lat_in.value = DEFAULTS["latency_seconds"]
    candle_in.value = DEFAULTS["candle_interval_seconds"]
    move_in.value = DEFAULTS["movement_threshold"]
    win_in.value = DEFAULTS["pattern_window_seconds"]
    state.push_log("UI: Reset inputs to strategy defaults")


def on_start_clicked() -> None:
    cfg = read_cfg_from_inputs()
    start_worker(ws_url=str(ws_url_in.value), cfg=cfg)
    state.push_log(f"UI: Started engine with ws_url={ws_url_in.value}")


def on_stop_clicked() -> None:
    stop_worker()
    state.push_log("UI: Stop requested")


def on_export_csv() -> None:
    with state.lock:
        eng = state.engine
    if not eng:
        export_lbl.set_text("Export: engine not running")
        return
    out = os.path.join(export_dir(), f"dashboard_trades_{now_ts()}.csv")
    path = eng.export_trades_csv(out)
    state.last_export_path = path
    export_lbl.set_text(f"Exported CSV: {path}")


def on_export_xlsx() -> None:
    with state.lock:
        eng = state.engine
    if not eng:
        export_lbl.set_text("Export: engine not running")
        return
    out = os.path.join(export_dir(), f"dashboard_trades_{now_ts()}.xlsx")
    path = eng.export_trades_xlsx(out)
    state.last_export_path = path
    export_lbl.set_text(f"Exported XLSX: {path}")



# Note: Button handlers are now wired up inside render_strategy_settings() and render_replay_controls()



def on_cancel_pending() -> None:
    with state.lock:
        eng = state.engine
    if not eng:
        state.push_log("UI: Cancel pending failed (engine not running)")
        return
    ok = eng.cancel_pending()
    state.push_log(f"UI: Cancel pending -> {ok}")


def on_manual_exit() -> None:
    with state.lock:
        eng = state.engine
    if not eng:
        state.push_log("UI: Manual exit failed (engine not running)")
        return
    ok = eng.manual_exit_active(reason="MANUAL_EXIT")
    state.push_log(f"UI: Manual exit -> {ok}")


cancel_btn.on("click", lambda e: on_cancel_pending())
exit_btn.on("click", lambda e: on_manual_exit())


def refresh_history() -> None:
    with state.lock:
        store = state.store
    if not store:
        # show empty but stable
        hist_table.columns = [{"name": "info", "label": "info", "field": "info"}]
        hist_table.rows = [{"info": "DB not open (start engine first)"}]
        return
    limit = int(hist_limit.value or 500)
    # If a range is set, use it; else show latest.
    sdate = (hist_start_date.value or "").strip()
    edate = (hist_end_date.value or "").strip()
    stime = (hist_start_time.value or "09:15").strip()
    etime = (hist_end_time.value or "15:30").strip()

    rows = None
    if sdate and edate:
        rng = ist_range_to_utc_iso(sdate, stime, edate, etime)
        if rng:
            rows = store.list_trades_range(rng["start_utc"], rng["end_utc"], limit=limit)
        else:
            state.push_log("History: invalid IST range format (use YYYY-MM-DD and HH:MM)")
            rows = []
    else:
        rows = store.list_trades(limit=limit)
    if not rows:
        hist_table.columns = [{"name": "info", "label": "info", "field": "info"}]
        hist_table.rows = [{"info": "No trades yet"}]
        return

    # Present a clean "trade history" view
    hist_table.columns = [
        {"name": "Order ID", "label": "Order ID", "field": "Order ID", "sortable": True},
        {"name": "Instrument", "label": "Instrument", "field": "Instrument", "sortable": True},
        {"name": "Entry (IST)", "label": "Entry (IST)", "field": "Entry (IST)", "sortable": True},
        {"name": "Entry Price", "label": "Entry Price", "field": "Entry Price", "sortable": True},
        {"name": "Exit (IST)", "label": "Exit (IST)", "field": "Exit (IST)", "sortable": True},
        {"name": "Exit Price", "label": "Exit Price", "field": "Exit Price", "sortable": True},
        {"name": "Charges", "label": "Charges", "field": "Charges", "sortable": True},
        {"name": "Net P&L", "label": "Net P&L", "field": "Net P&L", "sortable": True},
        {"name": "Reason", "label": "Reason", "field": "Reason", "sortable": True},
        {"name": "Patterns", "label": "Patterns", "field": "Patterns", "sortable": False},
    ]
    # Colorize Net P&L in history
    hist_table.add_slot(
        "body-cell-Net P&L",
        r"""
<q-td :props="props">
  <span :style="{color: (parseFloat(String(props.value).replace(/,/g,'')) >= 0 ? '#2ecc71' : '#ff4d4f'), fontWeight: '600'}">
    {{ props.value }}
  </span>
</q-td>
""",
    )

    view_rows = []
    for r in rows:
        entry_ts = parse_ts_utc(r.get("entry_time_utc"))
        exit_ts = parse_ts_utc(r.get("exit_time_utc"))
        view_rows.append(
            {
                "Order ID": r.get("order_id"),
                "Instrument": r.get("option_symbol"),
                "Entry (IST)": to_ist_str(entry_ts),
                "Entry Price": fmt_num(r.get("entry_price"), 2),
                "Exit (IST)": to_ist_str(exit_ts),
                "Exit Price": fmt_num(r.get("exit_price"), 2),
                # This is total transaction cost (brokerage+taxes) as per Angel One formula
                "Charges": fmt_money(r.get("transaction_cost")),
                "Net P&L": fmt_money(r.get("net_pnl")),
                "Reason": r.get("exit_reason") or "",
                "Patterns": r.get("patterns") or "",
            }
        )
    hist_table.rows = view_rows


hist_refresh.on("click", lambda e: refresh_history())


def set_default_history_range_from_feed() -> None:
    """Convenience: sets start/end date to feed-day if available."""
    snap = compute_active_pnl_snapshot()
    feed_ts = snap.get("feed_ts_utc")
    if not feed_ts:
        return
    if feed_ts.tzinfo is None:
        feed_ts = feed_ts.replace(tzinfo=timezone.utc)
    ist_dt = feed_ts.astimezone(IST)
    day = ist_dt.strftime("%Y-%m-%d")
    hist_start_date.value = day
    hist_end_date.value = day


def clear_history_filter() -> None:
    hist_start_date.value = ""
    hist_end_date.value = ""
    refresh_history()


hist_apply_range.on("click", lambda e: refresh_history())
hist_clear_range.on("click", lambda e: clear_history_filter())


def delete_history_range() -> None:
    with state.lock:
        store = state.store
    if not store:
        state.push_log("History delete: DB not open")
        return
    sdate = (hist_start_date.value or "").strip()
    edate = (hist_end_date.value or "").strip()
    stime = (hist_start_time.value or "09:15").strip()
    etime = (hist_end_time.value or "15:30").strip()
    if not (sdate and edate):
        state.push_log("History delete: set Start date and End date first")
        return
    rng = ist_range_to_utc_iso(sdate, stime, edate, etime)
    if not rng:
        state.push_log("History delete: invalid IST range format")
        return

    def do_delete() -> None:
        deleted = store.delete_trades_range(rng["start_utc"], rng["end_utc"])
        state.push_log(f"History delete: deleted {deleted} trades in range {sdate} {stime} -> {edate} {etime} IST")
        refresh_history()

    # NiceGUI provides ui.notify; use a simple confirm dialog
    dialog = ui.dialog()
    with dialog, ui.card():
        ui.label("Delete trades in selected IST range?")
        ui.label(f"{sdate} {stime}  →  {edate} {etime} (IST)")
        with ui.row():
            ui.button("Cancel", on_click=dialog.close)
            ui.button("Delete", color="negative", on_click=lambda: (do_delete(), dialog.close()))
    dialog.open()


hist_delete_range.on("click", lambda e: delete_history_range())


def refresh_ui() -> None:
    snap = compute_active_pnl_snapshot()

    if not state.running:
        conn_lbl.set_text("Status: STOPPED")
    else:
        conn = "CONNECTED" if snap.get("connected") else "DISCONNECTED"
        age = snap.get("last_msg_age")
        age_s = f"{age:.1f}s" if isinstance(age, (int, float)) and age is not None else "-"
        conn_lbl.set_text(f"Status: {conn} | last_msg_age={age_s} | reconnects={snap.get('reconnect_events')}")

    feed_lbl.set_text(f"Feed time (IST): {snap.get('feed_ts_ist') or '-'}")
    spot = snap.get("spot")
    spot_lbl.set_text(f"Spot: {spot:.2f}" if isinstance(spot, (int, float)) else "Spot: -")
    sym_lbl.set_text(f"Symbols: {snap.get('symbols')}")

    warm = snap.get("warmup") or {}
    if isinstance(warm, dict):
        if warm.get("ready"):
            sym_lbl.set_text(f"Symbols: {snap.get('symbols')} | Warm-up: READY")
        else:
            cov = warm.get("coverage_s") or 0.0
            need = warm.get("need_window_s") or 60.0
            candles = warm.get("candles_closed") or 0
            sym_lbl.set_text(
                f"Symbols: {snap.get('symbols')} | Warm-up: {cov:.0f}/{need:.0f}s window, candles={candles}"
            )

    # Active trade panel
    trade_state = snap.get("trade_state")
    if trade_state == "PENDING":
        t: PaperTrade = snap["trade"]
        trade_lbl.set_text(
            f"PENDING | id={t.order_id} trend={t.trend_number} {t.predicted_direction} | {t.option_symbol} | entry_at={to_ist_str(t.planned_entry_time_utc)} IST"
        )
    elif trade_state == "ACTIVE":
        t: PaperTrade = snap["trade"]
        cur = snap.get("cur")
        run_net = snap.get("run_net")
        run_pts = snap.get("run_points")
        tg = snap.get("tg")
        sl = snap.get("sl")
        trade_lbl.set_text(
            f"ACTIVE | id={t.order_id} | {t.option_symbol} | entry={t.entry_price:.2f} cur={(cur or 0):.2f} | TG={(tg or 0):.2f} SL={(sl or 0):.2f} | run_pts={(run_pts or 0):.2f} run_net={(run_net or 0):.2f} | patterns={t.patterns_str}"
        )
    else:
        trade_lbl.set_text("No active trade")

    # Positions UI (open/closed for feed-day)
    try:
        open_rows = positions_open_rows(snap)
        open_positions_table.rows = open_rows
        if open_rows:
            try:
                open_total = float(str(open_rows[0].get("P&L", "0")).replace(",", ""))
            except Exception:
                open_total = 0.0
        else:
            open_total = 0.0
        open_total_lbl.set_text(f"Open P&L: {open_total:,.2f}")

        closed_rows = closed_today_rows(snap.get("feed_ts_utc"), limit=500)
        closed_table.rows = closed_rows
        closed_net = closed_today_total_net(closed_rows)
        closed_total_lbl.set_text(f"Closed P&L: {closed_net:,.2f}")
    except Exception:
        pass

    # logs - All logs
    with state.lock:
        logs = list(state.logs)
        eng = state.engine
    
    # For Live Mode, read from log file (throttled to every 30 seconds)
    if state.mode == "live":
        import time
        current_time = time.time()
        
        # Only read log file every 30 seconds (matching candle interval)
        if current_time - state.last_log_read_time >= 30.0:
            state.last_log_read_time = current_time
            
            live_log_path = os.path.join(PROJECT_ROOT, "logs", "live_trading_engine.log")
            try:
                if os.path.exists(live_log_path):
                    with open(live_log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read last 200 lines from file
                        file_lines = f.readlines()
                        # Keep all non-empty lines INCLUDING separator lines (dashes)
                        live_logs = [line.strip() for line in file_lines[-200:] if line.strip()]
                        
                        # Display file logs in All Logs tab
                        log_box.clear()
                        for line in live_logs:
                            log_box.push(line)
                        
                        # Filter for pattern logs - Shows ongoing candle analysis and calculations
                        pattern_log_box.clear()
                        pattern_keywords = [
                            "candle closed", "direction changed", "direction =", 
                            "cumulative move", "threshold", "pctchange =", 
                            "start tracking", "not crossed yet",
                            "open=", "high=", "low=", "close="
                        ]
                        include_next_lines = 0
                        for line in live_logs:
                            line_lower = line.lower()
                            
                            if any(keyword in line_lower for keyword in pattern_keywords):
                                pattern_log_box.push(line)
                                include_next_lines = 5
                            elif include_next_lines > 0:
                                pattern_log_box.push(line)
                                include_next_lines -= 1
                        
                        # Filter for signal logs - Shows ONLY when signals are generated
                        signal_log_box.clear()
                        signal_keywords = [
                            "signal generated", "threshold crossed",
                            "iv pattern", "delta pattern", "volume ratio pattern", "premium pattern",
                            "call iv", "put iv", "call delta", "put delta", 
                            "call premium", "put premium", "patterns detected"
                        ]
                        include_next_lines = 0
                        for line in live_logs:
                            line_lower = line.lower()
                            
                            if any(keyword in line_lower for keyword in signal_keywords):
                                signal_log_box.push(line)
                                include_next_lines = 5
                            elif include_next_lines > 0:
                                signal_log_box.push(line)
                                include_next_lines -= 1
                else:
                    # No log file yet - show message (only update once)
                    if log_box._props.get('lines', []) == []:
                        log_box.clear()
                        log_box.push("⏳ Waiting for Live Mode logs... (Start Paper Trading Engine to see logs)")
                        pattern_log_box.clear()
                        signal_log_box.clear()
            except Exception as e:
                log_box.clear()
                log_box.push(f"❌ Error reading log file: {e}")
    else:
        # Replay Mode - use in-memory logs (update every refresh)
        log_box.clear()
        for line in logs[-200:]:
            log_box.push(line)
        
        # Pattern Analysis logs - filter for pattern-related messages
        pattern_log_box.clear()
        pattern_keywords = [
            "pattern", "indicator", "iv", "delta", "volume", "premium", 
            "bullish", "bearish", "entry", "exit", "option", "ce", "pe",
            "atm", "strike", "greek", "vega", "theta", "gamma"
        ]
        for line in logs[-200:]:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in pattern_keywords):
                pattern_log_box.push(line)
        
        # Signal Generation logs - filter for signal/trend-related messages
        signal_log_box.clear()
        signal_keywords = [
            "trend", "signal", "candle", "movement", "threshold", 
            "up", "down", "detected", "direction", "momentum",
            "price", "nifty", "spot", "change", "analysis"
        ]
        for line in logs[-200:]:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in signal_keywords):
                signal_log_box.push(line)

    # also keep history tab fresh-ish when open (cheap)
    try:
        if tabs.value == tab_history:
            refresh_history()
    except Exception:
        pass


ui.timer(1.0, refresh_ui)


def run() -> None:
    # NiceGUI entrypoint
    ui.run(host="127.0.0.1", port=8080, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    run()


