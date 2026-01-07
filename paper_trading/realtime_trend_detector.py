"""
Realtime Trend Detector (DB polling)

Reads NIFTY 50 spot ticks from SQLite `ltp_ticks` (written by the VPS/local data collector)
and builds 30-second OHLC candles. It emits a trend signal the moment a cumulative move
in a same-direction candle run crosses the movement threshold (default 0.11%).

This is real-time safe: it only uses candles as they close; it never retroactively labels
trend starts using future data.
"""

from __future__ import annotations

import os
import sys
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

# When imported from a script under paper_trading/, ensure project root is on sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse battle-tested candle & trend logic already in the repo.
from live_trading_engine import Candle, CandleBuilder, TrendSignalDetector  # type: ignore


logger = logging.getLogger(__name__)


def _parse_ts_utc(ts: Any) -> Optional[datetime]:
    """Parse DB timestamp into timezone-aware UTC datetime."""
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


@dataclass(frozen=True)
class TrendSignal:
    trend_number: int
    signal_time_utc: datetime
    trend_direction: str  # "UP" | "DOWN"
    spot_price: float
    cumulative_pct: float
    candle: Candle


class RealtimeTrendDetector:
    def __init__(
        self,
        *,
        spot_symbol: str = "NIFTY 50",
        candle_interval_seconds: int = 30,
        movement_threshold_pct: float = 0.11,
    ) -> None:
        self.spot_symbol = spot_symbol
        self._candle_builder = CandleBuilder(interval_seconds=candle_interval_seconds)
        self._detector = TrendSignalDetector(movement_threshold_pct=movement_threshold_pct)
        self._last_seen_id: int = 0
        self._candles_closed: int = 0

    def prime_from_latest(self, conn: sqlite3.Connection, *, lookback_rows: int = 500) -> None:
        """
        Set last_seen_id so we start near the latest spot ticks instead of replaying
        the entire DB history.
        """
        try:
            row = conn.execute(
                "SELECT MAX(id) FROM ltp_ticks WHERE symbol = ?",
                (self.spot_symbol,),
            ).fetchone()
            max_id = int(row[0] or 0) if row else 0
            lb = max(0, int(lookback_rows))
            self._last_seen_id = max(0, max_id - lb)
            logger.info("TrendDetector primed: spot_symbol=%s max_id=%s start_id=%s", self.spot_symbol, max_id, self._last_seen_id)
        except Exception as e:
            logger.warning("TrendDetector prime_from_latest failed: %s", e)

    @property
    def last_seen_id(self) -> int:
        return int(self._last_seen_id)

    @property
    def candles_closed(self) -> int:
        return int(self._candles_closed)

    def poll_new_spot_ticks(
        self,
        conn: sqlite3.Connection,
        *,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Incrementally fetch new spot ticks using `id > last_seen_id`.
        Returns ordered rows (oldest->newest).
        """
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, symbol, ltp, token
            FROM ltp_ticks
            WHERE symbol = ?
              AND id > ?
              AND ltp IS NOT NULL
              AND ltp > 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (self.spot_symbol, self._last_seen_id, int(limit)),
        )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for (row_id, ts, symbol, ltp, token) in rows:
            out.append({"id": row_id, "ts": ts, "symbol": symbol, "ltp": ltp, "token": token})
        if out:
            self._last_seen_id = int(out[-1]["id"])
        return out

    def process_ticks(self, ticks: Iterable[Dict[str, Any]]) -> List[TrendSignal]:
        """
        Feed ticks into the 30s candle builder. When a candle closes, push it to the
        detector and emit a TrendSignal if threshold is crossed.
        """
        signals: List[TrendSignal] = []
        for t in ticks:
            ts = _parse_ts_utc(t.get("ts"))
            price = t.get("ltp")
            if ts is None:
                continue
            try:
                px = float(price)
            except Exception:
                continue

            closed = self._candle_builder.update(ts, px)
            if not closed:
                continue

            self._candles_closed += 1
            info = self._detector.on_candle(closed)
            logger.debug(
                "CANDLE close=%s o=%.2f h=%.2f l=%.2f c=%.2f dir=%s",
                closed.ts.isoformat(),
                closed.open,
                closed.high,
                closed.low,
                closed.close,
                closed.direction,
            )
            if not info:
                continue

            try:
                sig = TrendSignal(
                    trend_number=int(info["trend_number"]),
                    signal_time_utc=info["signal_time"].replace(tzinfo=timezone.utc)
                    if info["signal_time"].tzinfo is None
                    else info["signal_time"],
                    trend_direction=str(info["trend_direction"]),
                    spot_price=float(info["spot_price"]),
                    cumulative_pct=float(info["cumulative_pct"]),
                    candle=info["candle"],
                )
                signals.append(sig)
                logger.info(
                    "TREND SIGNAL #%s %s | spot=%.2f | cum=%.4f%% | time=%s",
                    sig.trend_number,
                    sig.trend_direction,
                    sig.spot_price,
                    sig.cumulative_pct,
                    sig.signal_time_utc.isoformat(),
                )
            except Exception:
                continue

        return signals


def open_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


