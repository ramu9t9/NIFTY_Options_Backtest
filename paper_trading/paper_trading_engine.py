"""
Paper Trading Engine (Hybrid)

- Trend detection: DB polling (5 sec) on spot symbol 'NIFTY 50'
- Pattern analysis: DB query for last 60 seconds of option ticks
- Entry: DB LTP (at/after signal time)
- Exit monitoring: Angel One WebSocket for the active option token (Target/SL real-time)

Single process. One active trade at a time.
"""

from __future__ import annotations

import sys
import argparse
import csv
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pandas as pd

# When running as `py paper_trading\\paper_trading_engine.py`, Python sets sys.path[0]
# to the script directory (paper_trading/). Add project root so we can import
# sibling modules like `live_trading_engine.py`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from live_trading_engine import IST, calculate_transaction_cost, to_ist_str  # type: ignore

from paper_trading.realtime_pattern_analyzer import RealtimePatternAnalyzer, TradeSignal
from paper_trading.realtime_trend_detector import RealtimeTrendDetector, open_sqlite
from paper_trading.websocket_handler import AngelCredentials, AngelWebSocketHandler

logger = logging.getLogger(__name__)


def _parse_ts_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        out = dt.to_pydatetime()
        if out.tzinfo is None:
            out = out.replace(tzinfo=timezone.utc)
        return out
    except Exception:
        return None


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p) or "."
    os.makedirs(d, exist_ok=True)


@dataclass
class ActiveTrade:
    order_id: int
    trend_number: int
    option_symbol: str
    option_token: str
    entry_time_utc: datetime
    entry_price: float
    target_price: float
    stop_price: float
    max_exit_time_utc: datetime
    spot_price: float
    predicted_direction: str
    trend_direction: str
    patterns_count: int
    exit_time_utc: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


class PaperTradingEngine:
    def __init__(
        self,
        *,
        db_path: str,
        poll_seconds: float = 5.0,
        candle_seconds: int = 30,
        movement_threshold_pct: float = 0.11,
        pattern_window_seconds: int = 60,
        target_pct: float = 10.0,
        stop_pct: float = 5.0,
        max_hold_seconds: int = 180,
        lot_size: int = 3750,
        out_csv: Optional[str] = None,
        angel_creds: Optional[AngelCredentials] = None,
        start_lookback_rows: int = 500,
        start_from_beginning: bool = False,
        poll_log_seconds: float = 30.0,
    ) -> None:
        self.db_path = db_path
        self.poll_seconds = float(poll_seconds)
        self.target_pct = float(target_pct)
        self.stop_pct = float(stop_pct)
        self.max_hold_seconds = int(max_hold_seconds)
        self.lot_size = int(lot_size)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_csv = out_csv or os.path.join(os.path.dirname(__file__), f"paper_trades_{ts}.csv")

        self._conn = open_sqlite(db_path)
        self._conn.execute("PRAGMA busy_timeout=5000;")

        self._trend = RealtimeTrendDetector(
            spot_symbol="NIFTY 50",
            candle_interval_seconds=int(candle_seconds),
            movement_threshold_pct=float(movement_threshold_pct),
        )
        if not start_from_beginning:
            self._trend.prime_from_latest(self._conn, lookback_rows=int(start_lookback_rows))
        self._patterns = RealtimePatternAnalyzer(pattern_window_seconds=int(pattern_window_seconds))

        self._lock = threading.RLock()
        self._active: Optional[ActiveTrade] = None
        self._order_id = 0
        self._last_ws_tick_utc: Optional[datetime] = None
        self._last_feed_ts_utc: Optional[datetime] = None
        self._use_ws_exits: bool = True

        self._ws = AngelWebSocketHandler(
            creds=angel_creds,
            on_tick=self._on_ws_tick,
            on_status=lambda s: logger.info("%s", s),
            reconnect=True,
        )

        self._stop = threading.Event()
        self._csv_initialized = False
        self._poll_log_seconds = float(poll_log_seconds)
        self._last_poll_log_t = 0.0

    def close(self) -> None:
        try:
            self._ws.stop()
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass

    # ---------- DB helpers ----------

    def _get_option_entry_from_db(self, option_symbol: str, signal_time_utc: datetime) -> Optional[Dict[str, Any]]:
        """
        Find the first tick at/after signal time for entry.
        """
        q = """
        SELECT ts, symbol, token, ltp
        FROM ltp_ticks
        WHERE symbol = ?
          AND ts >= ?
          AND ltp IS NOT NULL
          AND ltp > 0
        ORDER BY id ASC
        LIMIT 1
        """
        row = self._conn.execute(q, (option_symbol, signal_time_utc.astimezone(timezone.utc).isoformat())).fetchone()
        if not row:
            return None
        return {"ts": row[0], "symbol": row[1], "token": row[2], "ltp": row[3]}

    def _get_option_ltp_at_or_before(self, option_symbol: str, ts_utc: datetime) -> Optional[float]:
        """
        Best-effort price for TIME_EXIT when WS isn't available:
        use last known DB tick at/before ts_utc.
        """
        q = """
        SELECT ltp
        FROM ltp_ticks
        WHERE symbol = ?
          AND ts <= ?
          AND ltp IS NOT NULL
          AND ltp > 0
        ORDER BY id DESC
        LIMIT 1
        """
        row = self._conn.execute(q, (option_symbol, ts_utc.astimezone(timezone.utc).isoformat())).fetchone()
        if not row:
            return None
        try:
            return float(row[0])
        except Exception:
            return None

    # ---------- CSV ----------

    def _csv_headers(self) -> list[str]:
        return [
            "Order ID",
            "Entry Date",
            "Entry Time",
            "Enter Price",
            "Exit Date",
            "Exit Time",
            "Exit Price",
            "Pnl Points",
            "gross_pnl",
            "transaction_cost",
            "net_pnl",
            "hold_time_minutes",
            "target_hit",
            "predicted_direction",
            "trend_direction",
            "signal_time_utc",
            "spot_price",
            "option_symbol",
            "patterns_count",
            "exit_reason",
        ]

    def _append_trade_csv(self, t: ActiveTrade) -> None:
        if t.exit_time_utc is None or t.exit_price is None or t.exit_reason is None:
            return

        entry = float(t.entry_price)
        exitp = float(t.exit_price)
        points = exitp - entry
        gross = points * float(self.lot_size)
        buy_val = entry * float(self.lot_size)
        sell_val = exitp * float(self.lot_size)
        tc = calculate_transaction_cost(buy_val, sell_val)
        net = gross - tc
        hold_min = (t.exit_time_utc - t.entry_time_utc).total_seconds() / 60.0
        target_hit = 1 if t.exit_reason == "TARGET" else 0

        _ensure_dir(self.out_csv)
        write_header = not os.path.exists(self.out_csv) or not self._csv_initialized
        with open(self.out_csv, "a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=self._csv_headers())
            if write_header:
                w.writeheader()
                self._csv_initialized = True
            w.writerow(
                {
                    "Order ID": t.order_id,
                    "Entry Date": t.entry_time_utc.astimezone(IST).strftime("%Y-%m-%d"),
                    "Entry Time": t.entry_time_utc.astimezone(IST).strftime("%H:%M:%S"),
                    "Enter Price": round(entry, 2),
                    "Exit Date": t.exit_time_utc.astimezone(IST).strftime("%Y-%m-%d"),
                    "Exit Time": t.exit_time_utc.astimezone(IST).strftime("%H:%M:%S"),
                    "Exit Price": round(exitp, 2),
                    "Pnl Points": round(points, 4),
                    "gross_pnl": round(gross, 2),
                    "transaction_cost": round(tc, 2),
                    "net_pnl": round(net, 2),
                    "hold_time_minutes": round(hold_min, 4),
                    "target_hit": target_hit,
                    "predicted_direction": t.predicted_direction,
                    "trend_direction": t.trend_direction,
                    "signal_time_utc": t.entry_time_utc.astimezone(timezone.utc).isoformat(),
                    "spot_price": round(float(t.spot_price), 2),
                    "option_symbol": t.option_symbol,
                    "patterns_count": t.patterns_count,
                    "exit_reason": t.exit_reason,
                }
            )

        logger.info(
            "TRADE DONE #%s | %s | %.2f -> %.2f | reason=%s | net=%.2f | hold=%.2f min | out=%s",
            t.order_id,
            t.option_symbol,
            entry,
            exitp,
            t.exit_reason,
            net,
            hold_min,
            self.out_csv,
        )

    # ---------- WS tick callback ----------

    def _on_ws_tick(self, tick: Dict[str, Any]) -> None:
        token = str(tick.get("token") or "")
        ltp = tick.get("ltp")
        ts_utc = tick.get("ts_utc")
        if not token or ltp is None:
            return
        if not isinstance(ts_utc, datetime):
            ts_utc = datetime.now(timezone.utc)

        with self._lock:
            self._last_ws_tick_utc = ts_utc
            t = self._active
            if not t or token != t.option_token:
                return

            cur = float(ltp)
            if cur >= t.target_price:
                self._finalize_exit(ts_utc, cur, "TARGET")
                return
            if cur <= t.stop_price:
                self._finalize_exit(ts_utc, cur, "STOP_LOSS")
                return

    # ---------- trade lifecycle ----------

    def _enter_trade(self, sig: TradeSignal) -> None:
        entry_row = self._get_option_entry_from_db(sig.option_symbol, sig.signal_time_utc)
        if not entry_row:
            logger.info("ENTRY skip: no option tick at/after signal for %s", sig.option_symbol)
            return

        ts = _parse_ts_utc(entry_row.get("ts")) or datetime.now(timezone.utc)
        token = entry_row.get("token") or sig.option_token
        if not token:
            logger.info("ENTRY skip: missing token for %s", sig.option_symbol)
            return

        entry = float(entry_row.get("ltp") or 0.0)
        if entry <= 0:
            logger.info("ENTRY skip: invalid entry ltp for %s", sig.option_symbol)
            return

        target = entry * (1.0 + (self.target_pct / 100.0))
        stop = entry * (1.0 - (self.stop_pct / 100.0))
        max_exit = ts + timedelta(seconds=self.max_hold_seconds)

        with self._lock:
            if self._active is not None:
                return
            self._order_id += 1
            self._active = ActiveTrade(
                order_id=self._order_id,
                trend_number=sig.trend_number,
                option_symbol=sig.option_symbol,
                option_token=str(token),
                entry_time_utc=ts,
                entry_price=entry,
                target_price=target,
                stop_price=stop,
                max_exit_time_utc=max_exit,
                spot_price=sig.spot_price,
                predicted_direction=sig.predicted_direction,
                trend_direction=sig.trend_direction,
                patterns_count=len(sig.patterns),
            )

        logger.info(
            "ENTRY #%s | %s | entry=%.2f | tg=%.2f sl=%.2f | time=%s IST",
            self._order_id,
            sig.option_symbol,
            entry,
            target,
            stop,
            to_ist_str(ts),
        )

        # Subscribe to WS for active token (start WS lazily to avoid rate limits when no trade)
        if self._use_ws_exits:
            self._ws.start()
            self._ws.subscribe(str(token), exchange_type=2, mode=1)

    def _finalize_exit(self, ts_utc: datetime, exit_price: float, reason: str) -> None:
        with self._lock:
            t = self._active
            if not t or t.exit_time_utc is not None:
                return
            t.exit_time_utc = ts_utc
            t.exit_price = float(exit_price)
            t.exit_reason = reason
            self._active = None

        try:
            self._ws.unsubscribe()
        except Exception:
            pass
        # Stop WS when idle (avoid reconnect loops / access rate limits)
        try:
            self._ws.stop()
        except Exception:
            pass

        self._append_trade_csv(t)

    def _check_time_exit(self, now_ts: Optional[datetime] = None) -> None:
        """
        Timer-based exit (3 min max hold).
        Prefer last feed timestamp from DB spot ticks; then WS tick timestamp; fallback to system time.

        This avoids huge hold times when replaying historical DB data, because we use the DB feed clock.
        """
        with self._lock:
            t = self._active
            if not t:
                return
            now_ts = now_ts or self._last_feed_ts_utc or self._last_ws_tick_utc or datetime.now(timezone.utc)
            if now_ts < t.max_exit_time_utc:
                return

        # Compute exit price outside lock (DB query)
        px = self._get_option_ltp_at_or_before(t.option_symbol, now_ts) or float(t.entry_price)
        self._finalize_exit(now_ts, float(px), "TIME_EXIT")

    # ---------- main loop ----------

    def run_forever(self) -> None:
        logger.info("Engine started | db=%s | out=%s", self.db_path, self.out_csv)
        try:
            while not self._stop.is_set():
                # 1) spot ticks -> candles -> trend signals
                spot_ticks = self._trend.poll_new_spot_ticks(self._conn)
                # Heartbeat log (so it's clear we're alive even when no signals)
                now_t = time.time()
                if (now_t - self._last_poll_log_t) >= self._poll_log_seconds:
                    with self._lock:
                        active = self._active
                    last_feed_ist = to_ist_str(self._last_feed_ts_utc) if self._last_feed_ts_utc else "-"
                    logger.info(
                        "POLL | new_spot_ticks=%s | last_seen_id=%s | candles_closed=%s | last_feed_ist=%s | active=%s",
                        len(spot_ticks),
                        self._trend.last_seen_id,
                        self._trend.candles_closed,
                        last_feed_ist,
                        ("YES" if active else "NO"),
                    )
                    self._last_poll_log_t = now_t
                for st in spot_ticks:
                    # maintain feed clock (DB timestamps) per-tick (prevents big jumps on backlog)
                    self._last_feed_ts_utc = _parse_ts_utc(st.get("ts")) or self._last_feed_ts_utc

                    trend_signals = self._trend.process_ticks([st])

                    # 2) for each trend signal, analyze patterns, maybe enter (only if no active trade)
                    with self._lock:
                        has_active = self._active is not None
                    if not has_active and trend_signals:
                        for tr in trend_signals:
                            tsig = self._patterns.analyze(self._conn, tr)
                            if tsig:
                                self._enter_trade(tsig)
                                break

                    # 3) time-based exit check using feed clock
                    self._check_time_exit(now_ts=self._last_feed_ts_utc)

                time.sleep(self.poll_seconds)
        finally:
            self.close()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid paper trading engine (DB signals + Angel WS exits)")
    ap.add_argument("--db-path", required=True, help="Path to nifty_local.db (SQLite)")
    ap.add_argument("--poll-seconds", type=float, default=5.0, help="DB polling interval for trend detection")
    ap.add_argument("--candle-seconds", type=int, default=30, help="Spot candle interval")
    ap.add_argument("--movement-threshold", type=float, default=0.11, help="Cumulative move threshold (%)")
    ap.add_argument("--pattern-window-seconds", type=int, default=60, help="Option pattern window size")
    ap.add_argument("--target-pct", type=float, default=10.0, help="Target percent")
    ap.add_argument("--stop-pct", type=float, default=5.0, help="Stop loss percent")
    ap.add_argument("--max-hold-seconds", type=int, default=180, help="Max hold (seconds)")
    ap.add_argument("--lot-size", type=int, default=3750, help="Lot size for P&L")
    ap.add_argument("--out-csv", default=None, help="Output CSV (default: paper_trades_*.csv)")
    ap.add_argument("--start-lookback-rows", type=int, default=500, help="On startup, begin near latest spot ticks by setting last_seen_id=max_id-lookback")
    ap.add_argument("--from-beginning", action="store_true", help="Replay from DB beginning (NOT recommended for live)")
    ap.add_argument("--poll-log-seconds", type=float, default=30.0, help="Log a heartbeat every N seconds showing DB polling progress")
    ap.add_argument("--no-ws-exits", action="store_true", help="Disable Angel WS exits (useful for offline DB replay tests)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = ap.parse_args()

    setup_logging(args.log_level)

    eng = PaperTradingEngine(
        db_path=args.db_path,
        poll_seconds=args.poll_seconds,
        candle_seconds=args.candle_seconds,
        movement_threshold_pct=args.movement_threshold,
        pattern_window_seconds=args.pattern_window_seconds,
        target_pct=args.target_pct,
        stop_pct=args.stop_pct,
        max_hold_seconds=args.max_hold_seconds,
        lot_size=args.lot_size,
        out_csv=args.out_csv,
        start_lookback_rows=args.start_lookback_rows,
        start_from_beginning=bool(args.from_beginning),
        poll_log_seconds=float(args.poll_log_seconds),
    )
    if args.no_ws_exits:
        eng._use_ws_exits = False  # intentionally simple CLI toggle
    eng.run_forever()


if __name__ == "__main__":
    main()


