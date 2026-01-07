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
    d = export_dir()
    return os.path.join(d, "live_trades.db")


def default_market_db_path() -> str:
    # Centralize Data Centre DB (used for optional backfill)
    return r"G:\Projects\Centralize Data Centre\data\nifty_local.db"


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
    ui.label("NIFTY Live Paper Trading Dashboard (Broadcaster Feed)").classes("text-lg")


with ui.row().classes("w-full"):
    with ui.card().classes("w-full"):
        ui.label("Controls")
        ws_url_in = ui.input("WebSocket URL", value=DEFAULTS["ws_url"]).classes("w-full")

        with ui.row():
            lot_in = ui.number("Lot size", value=DEFAULTS["lot_size"], format="%.0f")
            tg_in = ui.number("Target %", value=DEFAULTS["target_pct"], format="%.2f")
            sl_in = ui.number("Stop %", value=DEFAULTS["stop_pct"], format="%.2f")
            hold_in = ui.number("Max hold (min)", value=DEFAULTS["max_hold_minutes"], format="%.2f")

        with ui.row():
            use_ba_in = ui.checkbox("Use bid/ask fills", value=DEFAULTS["use_bid_ask"])
            slip_in = ui.number("Slippage (pts)", value=DEFAULTS["slippage_points"], format="%.2f")
            lat_in = ui.number("Latency (s)", value=DEFAULTS["latency_seconds"], format="%.2f")

        with ui.row():
            candle_in = ui.number("Candle interval (s)", value=DEFAULTS["candle_interval_seconds"], format="%.0f")
            move_in = ui.number("Movement threshold (%)", value=DEFAULTS["movement_threshold"], format="%.3f")
            win_in = ui.number("Pattern window (s)", value=DEFAULTS["pattern_window_seconds"], format="%.0f")

        with ui.row():
            start_btn = ui.button("Connect + Start Paper Trading", color="primary")
            stop_btn = ui.button("Stop", color="negative")
            reset_btn = ui.button("Reset to Strategy Defaults", color="accent")
            export_csv_btn = ui.button("Export CSV", color="secondary")
            export_xlsx_btn = ui.button("Export Excel (.xlsx)", color="secondary")

        export_lbl = ui.label("")

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
        with ui.card().classes("w-full"):
            ui.label("Live Log (feed timestamps inside messages)")
            log_box = ui.log(max_lines=300).classes("w-full")


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


start_btn.on("click", lambda e: on_start_clicked())
stop_btn.on("click", lambda e: on_stop_clicked())
reset_btn.on("click", lambda e: reset_to_defaults())
export_csv_btn.on("click", lambda e: on_export_csv())
export_xlsx_btn.on("click", lambda e: on_export_xlsx())


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

    # logs
    with state.lock:
        logs = list(state.logs)
        eng = state.engine
    log_box.clear()
    for line in logs[-200:]:
        log_box.push(line)

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


