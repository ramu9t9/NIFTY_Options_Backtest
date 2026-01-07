"""
Realtime Pattern Analyzer (DB query)

Given a TrendSignal, query the last `pattern_window_seconds` of option ticks from SQLite,
calculate the fast indicators (IV/Delta/Volume/Premium momentum), detect patterns, and
emit a TradeSignal if the predicted direction matches the spot trend direction.

No lookahead: window is strictly [signal_time - window, signal_time).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from live_trading_engine import (  # type: ignore
    THRESHOLDS_DEFAULT,
    calculate_fast_indicators,
    detect_fast_patterns,
    get_atm_strike,
)

from paper_trading.realtime_trend_detector import TrendSignal

logger = logging.getLogger(__name__)


def _ts_to_db_iso(dt_utc: datetime) -> str:
    # DB stores ISO strings (often with timezone). Use isoformat with tz.
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class TradeSignal:
    trend_number: int
    signal_time_utc: datetime
    trend_direction: str  # UP/DOWN
    predicted_direction: str  # BULLISH/BEARISH
    spot_price: float
    strike: int
    option_type: str  # CE/PE
    option_symbol: str
    option_token: Optional[str]
    patterns: List[Dict[str, Any]]


class RealtimePatternAnalyzer:
    def __init__(
        self,
        *,
        pattern_window_seconds: int = 60,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pattern_window_seconds = int(pattern_window_seconds)
        self.thresholds = dict(thresholds or THRESHOLDS_DEFAULT)

    def fetch_option_window(
        self,
        conn: sqlite3.Connection,
        *,
        signal_time_utc: datetime,
    ) -> pd.DataFrame:
        """
        Fetch option ticks in [signal_time - window, signal_time).
        """
        end = signal_time_utc
        start = end - timedelta(seconds=self.pattern_window_seconds)
        q = """
        SELECT ts, symbol, token, ltp, volume, oi, iv, delta, gamma, theta, vega
        FROM ltp_ticks
        WHERE ts >= ?
          AND ts < ?
          AND symbol != 'NIFTY 50'
          AND (symbol LIKE '%CE' OR symbol LIKE '%PE')
          AND ltp IS NOT NULL
          AND ltp > 0
        ORDER BY ts ASC, symbol ASC
        """
        df = pd.read_sql_query(q, conn, params=(_ts_to_db_iso(start), _ts_to_db_iso(end)))
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df[df["ts"].notna()].copy()
        return df

    def _determine_predicted_direction(self, patterns: List[Dict[str, Any]]) -> str:
        bullish = 0
        bearish = 0
        for p in patterns:
            d = str(p.get("direction") or "").upper()
            if d == "BULLISH":
                bullish += 1
            elif d == "BEARISH":
                bearish += 1
        return "BULLISH" if bullish > bearish else "BEARISH"

    def _trend_dir_to_pred(self, trend_direction: str) -> str:
        # Spot trend direction to expected predicted direction
        return "BULLISH" if str(trend_direction).upper() == "UP" else "BEARISH"

    def resolve_atm_option_symbol(
        self,
        conn: sqlite3.Connection,
        *,
        strike: int,
        option_type: str,  # CE/PE
        signal_time_utc: datetime,
        lookback_seconds: int = 120,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve the actual option symbol + token present in DB around the signal time.
        This auto-detects the current expiry, avoiding hardcoding.
        """
        end = signal_time_utc
        start = end - timedelta(seconds=int(lookback_seconds))
        like_suffix = f"%{strike:05d}{option_type.upper()}"
        q = """
        SELECT symbol, token
        FROM ltp_ticks
        WHERE ts >= ?
          AND ts <= ?
          AND symbol LIKE ?
        ORDER BY id DESC
        LIMIT 1
        """
        row = conn.execute(q, (_ts_to_db_iso(start), _ts_to_db_iso(end), like_suffix)).fetchone()
        if not row:
            return None, None
        return str(row[0]), str(row[1]) if row[1] is not None else None

    def analyze(self, conn: sqlite3.Connection, trend: TrendSignal) -> Optional[TradeSignal]:
        df = self.fetch_option_window(conn, signal_time_utc=trend.signal_time_utc)
        if df.empty:
            logger.info("PATTERN skip: no option ticks in last %ss", self.pattern_window_seconds)
            return None

        metrics = calculate_fast_indicators(df)
        patterns = detect_fast_patterns(metrics, self.thresholds)
        if not patterns:
            logger.info("PATTERN none: trend #%s (%s) at %s", trend.trend_number, trend.trend_direction, trend.signal_time_utc.isoformat())
            return None

        predicted = self._determine_predicted_direction(patterns)
        expected = self._trend_dir_to_pred(trend.trend_direction)
        if predicted != expected:
            logger.info(
                "PATTERN mismatch: trend #%s %s expected=%s predicted=%s (patterns=%s)",
                trend.trend_number,
                trend.trend_direction,
                expected,
                predicted,
                ",".join([str(p.get("pattern_type")) for p in patterns]),
            )
            return None

        strike = get_atm_strike(float(trend.spot_price))
        opt_type = "CE" if trend.trend_direction.upper() == "UP" else "PE"
        sym, tok = self.resolve_atm_option_symbol(conn, strike=strike, option_type=opt_type, signal_time_utc=trend.signal_time_utc)
        if not sym:
            logger.info("PATTERN skip: could not resolve live option symbol for %s %s", strike, opt_type)
            return None

        logger.info(
            "TRADE SIGNAL: trend #%s %s | pred=%s | spot=%.2f | %s",
            trend.trend_number,
            trend.trend_direction,
            predicted,
            float(trend.spot_price),
            sym,
        )
        return TradeSignal(
            trend_number=trend.trend_number,
            signal_time_utc=trend.signal_time_utc,
            trend_direction=trend.trend_direction,
            predicted_direction=predicted,
            spot_price=float(trend.spot_price),
            strike=int(strike),
            option_type=opt_type,
            option_symbol=sym,
            option_token=tok,
            patterns=patterns,
        )


