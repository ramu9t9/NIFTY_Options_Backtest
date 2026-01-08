"""
Reusable live trading engine for:
- WebSocket broadcaster ticks (5s cadence)
- 30s candle building for NIFTY 50
- real-time trend signal detection (no lookahead)
- fast indicator pattern detection on last 60s (no lookahead)
- paper trade state machine (single active trade)

Used by:
- live_paper_trader.py (CLI)
- live_dashboard.py (NiceGUI)
"""

from __future__ import annotations

import csv
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from trade_store import TradeStore

IST = timezone(timedelta(hours=5, minutes=30))

logger = logging.getLogger(__name__)


def parse_ts_utc(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    try:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except Exception:
        return None


def to_ist_str(dt_utc: Optional[datetime], fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if not dt_utc:
        return ""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(IST).strftime(fmt)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v == 0 or pd.isna(v):
            return None
        return v
    except Exception:
        return None


def get_atm_strike(spot_price: float) -> int:
    return int(round(spot_price / 50.0) * 50)


def calculate_transaction_cost(buy_value: float, sell_value: float) -> float:
    # Angel One formula (same as backtest_strategy.py)
    brokerage = 40.00
    stt = sell_value * 0.0625 / 100
    total_turnover = buy_value + sell_value
    transaction_charges = total_turnover * 0.053 / 100
    gst = (brokerage + transaction_charges) * 18 / 100
    sebi = (total_turnover / 10_000_000) * 10
    stamp_duty = buy_value * 0.003 / 100
    return brokerage + stt + transaction_charges + gst + sebi + stamp_duty


# -----------------------
# Patterns (fast-only)
# -----------------------

THRESHOLDS_DEFAULT = {
    "iv_change_pct": 5.0,
    "volume_ratio_change": 10.0,
    "delta_change": 0.03,
    "premium_momentum": 2.0,
}


def calculate_fast_indicators(options_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate fast indicators from options data.
    Logs detailed calculations for debugging and understanding.
    """
    if options_df.empty:
        logger.info("PATTERN ANALYSIS | No options data available")
        return {}

    logger.info("=" * 60)
    logger.info("PATTERN ANALYSIS | Calculating indicators")
    logger.info("  Options ticks: %d", len(options_df))

    df = options_df.copy()
    df["option_type"] = df["symbol"].astype(str).apply(lambda x: "CALL" if x.endswith("CE") else "PUT")
    calls = df[df["option_type"] == "CALL"].copy()
    puts = df[df["option_type"] == "PUT"].copy()
    
    logger.info("  Calls: %d ticks, Puts: %d ticks", len(calls), len(puts))

    metrics: Dict[str, float] = {}

    # IV change %
    if not calls.empty and "iv" in calls.columns:
        g = calls.groupby("symbol", sort=False)
        first = g.first()["iv"].mean()
        last = g.last()["iv"].mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            change_pct = ((float(last) - float(first)) / float(first)) * 100.0
            metrics["call_iv_change_pct"] = change_pct
            logger.info("CALL IV CHANGE:")
            logger.info("  First IV (avg): %.2f%%", float(first))
            logger.info("  Last IV (avg): %.2f%%", float(last))
            logger.info("  Change = ((%.2f - %.2f) / %.2f) * 100 = %.2f%%", 
                       float(last), float(first), float(first), change_pct)

    if not puts.empty and "iv" in puts.columns:
        g = puts.groupby("symbol", sort=False)
        first = g.first()["iv"].mean()
        last = g.last()["iv"].mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            change_pct = ((float(last) - float(first)) / float(first)) * 100.0
            metrics["put_iv_change_pct"] = change_pct
            logger.info("PUT IV CHANGE:")
            logger.info("  First IV (avg): %.2f%%", float(first))
            logger.info("  Last IV (avg): %.2f%%", float(last))
            logger.info("  Change = ((%.2f - %.2f) / %.2f) * 100 = %.2f%%", 
                       float(last), float(first), float(first), change_pct)

    # Volume ratio change
    if not calls.empty and not puts.empty and "volume" in calls.columns:
        cg = calls.groupby("symbol", sort=False)
        pg = puts.groupby("symbol", sort=False)
        call_vol_first = float(cg.first()["volume"].sum() or 0)
        put_vol_first = float(pg.first()["volume"].sum() or 0)
        put_vol_last = float(pg.last()["volume"].sum() or 0)
        if call_vol_first > 0 and put_vol_first > 0:
            pc_first = put_vol_first / call_vol_first
            pc_last = put_vol_last / call_vol_first if call_vol_first > 0 else 0.0
            if pc_first > 0:
                change_pct = ((pc_last - pc_first) / pc_first) * 100.0
                metrics["pc_volume_ratio_change"] = change_pct
                logger.info("PUT/CALL VOLUME RATIO:")
                logger.info("  Call Volume: %.0f", call_vol_first)
                logger.info("  Put Volume First: %.0f", put_vol_first)
                logger.info("  Put Volume Last: %.0f", put_vol_last)
                logger.info("  PC Ratio First: %.4f", pc_first)
                logger.info("  PC Ratio Last: %.4f", pc_last)
                logger.info("  Change = ((%.4f - %.4f) / %.4f) * 100 = %.2f%%", 
                           pc_last, pc_first, pc_first, change_pct)

    # Delta change
    if not calls.empty and "delta" in calls.columns:
        g = calls.groupby("symbol", sort=False)
        first = g.first()["delta"].mean()
        last = g.last()["delta"].mean()
        if pd.notna(first) and pd.notna(last):
            change = float(last) - float(first)
            metrics["call_delta_change"] = change
            logger.info("CALL DELTA CHANGE:")
            logger.info("  First Delta (avg): %.4f", float(first))
            logger.info("  Last Delta (avg): %.4f", float(last))
            logger.info("  Change = %.4f - %.4f = %.4f", float(last), float(first), change)

    if not puts.empty and "delta" in puts.columns:
        g = puts.groupby("symbol", sort=False)
        first = g.first()["delta"].mean()
        last = g.last()["delta"].mean()
        if pd.notna(first) and pd.notna(last):
            change = float(last) - float(first)
            metrics["put_delta_change"] = change
            logger.info("PUT DELTA CHANGE:")
            logger.info("  First Delta (avg): %.4f", float(first))
            logger.info("  Last Delta (avg): %.4f", float(last))
            logger.info("  Change = %.4f - %.4f = %.4f", float(last), float(first), change)

    # Premium momentum (% change)
    if not calls.empty:
        g = calls.groupby("symbol", sort=False)
        first = g.first()["ltp"].mean()
        last = g.last()["ltp"].mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            momentum = ((float(last) - float(first)) / float(first)) * 100.0
            metrics["call_premium_momentum"] = momentum
            logger.info("CALL PREMIUM MOMENTUM:")
            logger.info("  First Premium (avg): %.2f", float(first))
            logger.info("  Last Premium (avg): %.2f", float(last))
            logger.info("  Momentum = ((%.2f - %.2f) / %.2f) * 100 = %.2f%%", 
                       float(last), float(first), float(first), momentum)

    if not puts.empty:
        g = puts.groupby("symbol", sort=False)
        first = g.first()["ltp"].mean()
        last = g.last()["ltp"].mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            momentum = ((float(last) - float(first)) / float(first)) * 100.0
            metrics["put_premium_momentum"] = momentum
            logger.info("PUT PREMIUM MOMENTUM:")
            logger.info("  First Premium (avg): %.2f", float(first))
            logger.info("  Last Premium (avg): %.2f", float(last))
            logger.info("  Momentum = ((%.2f - %.2f) / %.2f) * 100 = %.2f%%", 
                       float(last), float(first), float(first), momentum)

    logger.info("INDICATORS CALCULATED: %d metrics", len(metrics))
    return metrics


def detect_fast_patterns(metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Detect patterns from calculated metrics.
    Logs detailed threshold comparisons and pattern detection.
    """
    patterns: List[Dict[str, Any]] = []
    
    logger.info("-" * 60)
    logger.info("PATTERN DETECTION | Checking thresholds")

    # IV patterns
    if "call_iv_change_pct" in metrics:
        v = metrics["call_iv_change_pct"]
        threshold = thresholds["iv_change_pct"]
        detected = abs(v) > threshold
        logger.info("CALL IV PATTERN:")
        logger.info("  Value: %.2f%%, Threshold: %.2f%%", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO", 
                   f"(BEARISH)" if detected and v > 0 else f"(BULLISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "CALL_IV_CHANGE", "value": v, "direction": "BEARISH" if v > 0 else "BULLISH"})
            
    if "put_iv_change_pct" in metrics:
        v = metrics["put_iv_change_pct"]
        threshold = thresholds["iv_change_pct"]
        detected = abs(v) > threshold
        logger.info("PUT IV PATTERN:")
        logger.info("  Value: %.2f%%, Threshold: %.2f%%", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO",
                   f"(BULLISH)" if detected and v < 0 else f"(BEARISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "PUT_IV_CHANGE", "value": v, "direction": "BULLISH" if v < 0 else "BEARISH"})

    # Volume
    if "pc_volume_ratio_change" in metrics:
        v = metrics["pc_volume_ratio_change"]
        threshold = thresholds["volume_ratio_change"]
        detected = abs(v) > threshold
        logger.info("VOLUME RATIO PATTERN:")
        logger.info("  Value: %.2f%%, Threshold: %.2f%%", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO",
                   f"(BULLISH)" if detected and v < 0 else f"(BEARISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "PC_VOLUME_RATIO_CHANGE", "value": v, "direction": "BULLISH" if v < 0 else "BEARISH"})

    # Delta
    if "call_delta_change" in metrics:
        v = metrics["call_delta_change"]
        threshold = thresholds["delta_change"]
        detected = abs(v) > threshold
        logger.info("CALL DELTA PATTERN:")
        logger.info("  Value: %.4f, Threshold: %.4f", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO",
                   f"(BULLISH)" if detected and v > 0 else f"(BEARISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "CALL_DELTA_CHANGE", "value": v, "direction": "BULLISH" if v > 0 else "BEARISH"})
            
    if "put_delta_change" in metrics:
        v = metrics["put_delta_change"]
        threshold = thresholds["delta_change"]
        detected = abs(v) > threshold
        logger.info("PUT DELTA PATTERN:")
        logger.info("  Value: %.4f, Threshold: %.4f", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO",
                   f"(BEARISH)" if detected and v < 0 else f"(BULLISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "PUT_DELTA_CHANGE", "value": v, "direction": "BEARISH" if v < 0 else "BULLISH"})

    # Premium momentum
    if "call_premium_momentum" in metrics:
        v = metrics["call_premium_momentum"]
        threshold = thresholds["premium_momentum"]
        detected = abs(v) > threshold
        logger.info("CALL PREMIUM PATTERN:")
        logger.info("  Value: %.2f%%, Threshold: %.2f%%", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO",
                   f"(BULLISH)" if detected and v > 0 else f"(BEARISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "CALL_PREMIUM_MOMENTUM", "value": v, "direction": "BULLISH" if v > 0 else "BEARISH"})
            
    if "put_premium_momentum" in metrics:
        v = metrics["put_premium_momentum"]
        threshold = thresholds["premium_momentum"]
        detected = abs(v) > threshold
        logger.info("PUT PREMIUM PATTERN:")
        logger.info("  Value: %.2f%%, Threshold: %.2f%%", v, threshold)
        logger.info("  Detected: %s %s", "YES" if detected else "NO",
                   f"(BEARISH)" if detected and v > 0 else f"(BULLISH)" if detected else "")
        if detected:
            patterns.append({"pattern_type": "PUT_PREMIUM_MOMENTUM", "value": v, "direction": "BEARISH" if v > 0 else "BULLISH"})

    logger.info("=" * 60)
    logger.info("PATTERNS DETECTED: %d", len(patterns))
    if patterns:
        for i, p in enumerate(patterns, 1):
            logger.info("  %d. %s: %.4f (%s)", i, p["pattern_type"], p["value"], p["direction"])
    logger.info("=" * 60)

    return patterns


def determine_direction_from_patterns(patterns_detected: Dict[str, float]) -> str:
    bullish = 0
    bearish = 0
    for pattern_type, value in patterns_detected.items():
        if pattern_type == "PUT_IV_CHANGE" and value < 0:
            bullish += 1
        elif pattern_type == "CALL_IV_CHANGE" and value > 0:
            bearish += 1
        elif pattern_type == "PUT_PREMIUM_MOMENTUM" and value > 0:
            bearish += 1
        elif pattern_type == "CALL_PREMIUM_MOMENTUM" and value > 0:
            bullish += 1
        elif pattern_type == "PC_VOLUME_RATIO_CHANGE":
            if value < 0:
                bullish += 1
            else:
                bearish += 1
    return "BULLISH" if bullish > bearish else "BEARISH"


# -----------------------
# Spot candles + signals
# -----------------------


@dataclass
class Candle:
    ts: datetime  # bucket start (UTC)
    open: float
    high: float
    low: float
    close: float

    @property
    def pct_change(self) -> float:
        if self.open <= 0:
            return 0.0
        return ((self.close - self.open) / self.open) * 100.0

    @property
    def direction(self) -> int:
        pc = self.pct_change
        if pc > 0.01:
            return 1
        if pc < -0.01:
            return -1
        return 0


class CandleBuilder:
    def __init__(self, interval_seconds: int):
        self.interval_seconds = int(interval_seconds)
        self._cur_bucket: Optional[datetime] = None
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0

    def update(self, ts_utc: datetime, price: float) -> Optional[Candle]:
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)

        bucket = ts_utc.replace(microsecond=0)
        sec = bucket.second - (bucket.second % self.interval_seconds)
        bucket = bucket.replace(second=sec)

        if self._cur_bucket is None:
            self._start(bucket, price)
            return None

        if bucket == self._cur_bucket:
            self._high = max(self._high, price)
            self._low = min(self._low, price)
            self._close = price
            return None

        prev = Candle(ts=self._cur_bucket, open=self._open, high=self._high, low=self._low, close=self._close)
        self._start(bucket, price)
        return prev

    def _start(self, bucket: datetime, price: float) -> None:
        self._cur_bucket = bucket
        self._open = price
        self._high = price
        self._low = price
        self._close = price


class TrendSignalDetector:
    def __init__(self, movement_threshold_pct: float):
        self.movement_threshold_pct = float(movement_threshold_pct)
        self.current_direction: Optional[int] = None
        self.direction_start_price: Optional[float] = None
        self.already_signaled = False
        self.trend_counter = 0

    def on_candle(self, candle: Candle) -> Optional[Dict[str, Any]]:
        """
        Process a closed candle and detect trend signals.
        Logs detailed step-by-step logic for debugging and understanding.
        """
        d = candle.direction
        
        # Log candle details
        logger.info("=" * 60)
        logger.info("CANDLE CLOSED | Time: %s", candle.ts.isoformat())
        logger.info("  Open=%.2f, High=%.2f, Low=%.2f, Close=%.2f", 
                   candle.open, candle.high, candle.low, candle.close)
        
        # Calculate and log direction
        pct_change = ((candle.close - candle.open) / candle.open) * 100.0
        logger.info("  PctChange = ((Close - Open) / Open) * 100")
        logger.info("  PctChange = ((%.2f - %.2f) / %.2f) * 100 = %.4f%%", 
                   candle.close, candle.open, candle.open, pct_change)
        
        if pct_change > 0.01:
            logger.info("  Direction = 1 (UP) [YES]")
        elif pct_change < -0.01:
            logger.info("  Direction = -1 (DOWN) [YES]")
        else:
            logger.info("  Direction = 0 (NEUTRAL)")
        
        # Check for direction change
        if d != self.current_direction or d == 0:
            logger.info("DIRECTION CHANGED | From %s to %s", self.current_direction, d)
            logger.info("  Start tracking from: %.2f (candle open)", candle.open)
            logger.info("  Threshold: %.2f%%", self.movement_threshold_pct)
            logger.info("  NOT crossed yet [NO]")
            logger.info("--- End of Candle Cycle ---")
            logger.info("")  # Blank line 1
            logger.info("")  # Blank line 2
            logger.info("")  # Blank line 3
            logger.info("")  # Blank line 4
            
            self.current_direction = d
            self.direction_start_price = candle.open
            self.already_signaled = False
            return None

        if self.already_signaled or not self.direction_start_price:
            logger.info("CUMULATIVE MOVE | Already signaled or no start price")
            logger.info("--- End of Candle Cycle ---")
            logger.info("")  # Blank line 1
            logger.info("")  # Blank line 2
            logger.info("")  # Blank line 3
            logger.info("")  # Blank line 4
            return None

        # Calculate cumulative move
        cumulative_pct = ((candle.close - self.direction_start_price) / self.direction_start_price) * 100.0
        logger.info("CUMULATIVE MOVE | Same direction continues")
        logger.info("  Cumulative = ((Close - Start) / Start) * 100")
        logger.info("  Cumulative = ((%.2f - %.2f) / %.2f) * 100 = %.4f%%", 
                   candle.close, self.direction_start_price, self.direction_start_price, cumulative_pct)
        logger.info("  Threshold = %.2f%%", self.movement_threshold_pct)
        
        # Check threshold
        if abs(cumulative_pct) >= self.movement_threshold_pct:
            self.trend_counter += 1
            self.already_signaled = True
            
            logger.info("  THRESHOLD CROSSED! [YES]")
            logger.info("=" * 60)
            logger.info("SIGNAL GENERATED | Trend #%d", self.trend_counter)
            logger.info("  Entry Time: %s", candle.ts.isoformat())
            logger.info("  Entry Price: %.2f", candle.close)
            logger.info("  Direction: %s", "UP" if d == 1 else "DOWN")
            logger.info("  Cumulative Move: %.4f%%", cumulative_pct)
            logger.info("=" * 60)
            logger.info("--- End of Candle Cycle ---")
            logger.info("")  # Blank line 1
            logger.info("")  # Blank line 2
            logger.info("")  # Blank line 3
            logger.info("")  # Blank line 4
            
            return {
                "trend_number": self.trend_counter,
                "signal_time": candle.ts,
                "spot_price": candle.close,
                "trend_direction": "UP" if d == 1 else "DOWN",
                "cumulative_pct": cumulative_pct,
                "candle": candle,
            }
        else:
            logger.info("  NOT crossed yet [NO]")
            logger.info("--- End of Candle Cycle ---")
            logger.info("")  # Blank line 1
            logger.info("")  # Blank line 2
            logger.info("")  # Blank line 3
            logger.info("")  # Blank line 4
        
        return None


# -----------------------
# Trade state machine
# -----------------------


EXPIRY_RE = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})\d+(CE|PE)$")


def parse_expiry_str(expiry_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(expiry_str, "%d%b%y").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def pick_nearest_expiry(symbols: List[str], now_utc: datetime) -> Optional[str]:
    expiries: set[str] = set()
    for s in symbols:
        m = EXPIRY_RE.match(s)
        if m:
            expiries.add(m.group(1))
    best: Optional[Tuple[str, datetime]] = None
    for exp in expiries:
        dt = parse_expiry_str(exp)
        if not dt:
            continue
        if dt < now_utc:
            continue
        if best is None or dt < best[1]:
            best = (exp, dt)
    return best[0] if best else None


def fill_price(tick: Dict[str, Any], side: str, use_bid_ask: bool, slippage_points: float) -> Optional[float]:
    ltp_f = safe_float(tick.get("ltp"))
    bid_f = safe_float(tick.get("bid"))
    ask_f = safe_float(tick.get("ask"))

    if not use_bid_ask:
        px = ltp_f
    else:
        if side == "BUY":
            px = ask_f or ltp_f
        else:
            px = bid_f or ltp_f
    if px is None:
        return None
    if slippage_points and slippage_points > 0:
        if side == "BUY":
            px += float(slippage_points)
        else:
            px -= float(slippage_points)
    return px


@dataclass
class PaperTrade:
    order_id: int
    trend_number: int
    predicted_direction: str
    actual_direction: str
    option_symbol: str
    option_type: str
    strike: int
    expiry: str
    patterns_count: int
    patterns_str: str
    spot_price: float
    rally_start_utc: datetime

    signal_time_utc: datetime
    planned_entry_time_utc: datetime
    entry_time_utc: Optional[datetime] = None
    exit_time_utc: Optional[datetime] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    target_hit: bool = False

    gross_pnl: Optional[float] = None
    transaction_cost: Optional[float] = None
    net_pnl: Optional[float] = None
    pnl_points: Optional[float] = None
    hold_time_minutes: Optional[float] = None


@dataclass
class EngineConfig:
    lot_size: int = 3750
    target_pct: float = 10.0
    stop_pct: float = 5.0
    max_hold_minutes: float = 3.0
    use_bid_ask: bool = False
    slippage_points: float = 0.0
    latency_seconds: float = 0.0
    candle_interval_seconds: int = 30
    movement_threshold: float = 0.11
    pattern_window_seconds: int = 60
    thresholds: Dict[str, float] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.thresholds is None:
            self.thresholds = dict(THRESHOLDS_DEFAULT)


BackfillFn = Callable[[datetime, datetime], List[Dict[str, Any]]]


EngineLogFn = Callable[[str], None]


class LiveTradingEngine:
    """
    Thread-safe engine:
    - call on_tick() for each tick dict from broadcaster
    - reads symbols list (for expiry selection)
    - maintains single pending/active trade
    """

    def __init__(
        self,
        cfg: EngineConfig,
        log_fn: Optional[EngineLogFn] = None,
        trade_store: Optional[TradeStore] = None,
        backfill_fn: Optional[BackfillFn] = None,
    ) -> None:
        self.cfg = cfg
        self._log_fn = log_fn
        self._store = trade_store
        self._backfill_fn = backfill_fn
        self._lock = threading.RLock()

        self._candle_builder = CandleBuilder(interval_seconds=cfg.candle_interval_seconds)
        self._detector = TrendSignalDetector(movement_threshold_pct=cfg.movement_threshold)

        self._order_id = 0
        self._active_trade: Optional[PaperTrade] = None
        self._pending_trade: Optional[PaperTrade] = None
        self._completed: List[PaperTrade] = []

        self._option_ticks: List[Dict[str, Any]] = []
        self._latest_tick_by_symbol: Dict[str, Dict[str, Any]] = {}

        self._last_feed_ts_utc: Optional[datetime] = None
        self._last_spot: Optional[float] = None
        self._candles_closed: int = 0

    # ---------- state getters ----------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            active = self._active_trade
            pending = self._pending_trade
            warm = self.warmup_status()
            return {
                "last_feed_ts_utc": self._last_feed_ts_utc,
                "last_feed_ts_ist": to_ist_str(self._last_feed_ts_utc) if self._last_feed_ts_utc else "",
                "last_spot": self._last_spot,
                "active_trade": active,
                "pending_trade": pending,
                "completed_trades": list(self._completed),
                "warmup": warm,
            }

    def warmup_status(self) -> Dict[str, Any]:
        """
        Warm-up readiness:
        - needs at least 1 closed 30s candle
        - needs option tick coverage of pattern_window_seconds BEFORE signal time
        """
        with self._lock:
            last_ts = self._last_feed_ts_utc
            if not last_ts:
                return {"ready": False, "reason": "waiting_for_feed", "coverage_s": 0.0, "candles_closed": self._candles_closed}

            # option tick coverage in last window
            start = last_ts - timedelta(seconds=self.cfg.pattern_window_seconds)
            window_ticks = [x for x in self._option_ticks if start <= x["ts"] <= last_ts]
            if not window_ticks:
                return {
                    "ready": False,
                    "reason": "collecting_option_window",
                    "coverage_s": 0.0,
                    "candles_closed": self._candles_closed,
                    "need_window_s": float(self.cfg.pattern_window_seconds),
                }
            earliest = min(x["ts"] for x in window_ticks)
            coverage = (last_ts - earliest).total_seconds()
            ready = (coverage >= float(self.cfg.pattern_window_seconds)) and (self._candles_closed >= 1)
            return {
                "ready": ready,
                "reason": "ok" if ready else ("need_candle_close" if self._candles_closed < 1 else "collecting_option_window"),
                "coverage_s": float(coverage),
                "need_window_s": float(self.cfg.pattern_window_seconds),
                "candles_closed": self._candles_closed,
            }

    def has_open_position(self) -> bool:
        with self._lock:
            return self._active_trade is not None or self._pending_trade is not None

    def completed_trades(self) -> List[PaperTrade]:
        with self._lock:
            return list(self._completed)

    # ---------- trade management ----------
    def cancel_pending(self) -> bool:
        """Cancel a pending (not yet entered) trade."""
        with self._lock:
            if not self._pending_trade:
                return False
            t = self._pending_trade
            self._pending_trade = None
        self._emit(
            f"MANUAL CANCEL | id={t.order_id} trend={t.trend_number} | symbol={t.option_symbol} | time={to_ist_str(datetime.now(timezone.utc))} IST"
        )
        return True

    def manual_exit_active(self, reason: str = "MANUAL_EXIT") -> bool:
        """
        Force-exit the active trade using the latest known tick for that option symbol.
        Returns False if no active trade or no latest tick/price available.
        """
        with self._lock:
            t = self._active_trade
            if not t or not t.entry_price:
                return False
            tick = self._latest_tick_by_symbol.get(t.option_symbol)
            if not tick:
                return False
            cur_exit = fill_price(tick, "SELL", self.cfg.use_bid_ask, self.cfg.slippage_points)
            if cur_exit is None:
                ltp = safe_float(tick.get("ltp"))
                if ltp is None:
                    return False
                cur_exit = float(ltp)

        # finalize outside lock with normal exit pipeline
        now_utc = parse_ts_utc(tick.get("ts")) or datetime.now(timezone.utc)
        with self._lock:
            # re-check still active
            if self._active_trade is None or self._active_trade.order_id != t.order_id:
                return False
            # override exit_reason by temporarily setting config? We'll finalize then override.
            self._finalize_manual_exit(now_utc, float(cur_exit), reason)
        return True

    def _finalize_manual_exit(self, now_utc: datetime, cur_exit: float, reason: str) -> None:
        """Finalize exit similarly to _maybe_exit_trade, but with explicit reason."""
        t = self._active_trade
        if not t or t.entry_price is None or t.entry_time_utc is None:
            return

        entry = float(t.entry_price)
        hold_min = (now_utc - t.entry_time_utc).total_seconds() / 60.0

        t.exit_time_utc = now_utc
        t.exit_price = float(cur_exit)
        t.exit_reason = reason
        t.target_hit = False

        points = float(t.exit_price) - entry
        gross = points * float(self.cfg.lot_size)
        buy_val = entry * float(self.cfg.lot_size)
        sell_val = float(t.exit_price) * float(self.cfg.lot_size)
        tc = calculate_transaction_cost(buy_val, sell_val)
        net = gross - tc

        t.pnl_points = points
        t.gross_pnl = gross
        t.transaction_cost = tc
        t.net_pnl = net
        t.hold_time_minutes = hold_min

        self._completed.append(t)
        self._active_trade = None

        if self._store:
            try:
                self._store.insert_trade(
                    {
                        "order_id": t.order_id,
                        "trend_number": t.trend_number,
                        "predicted_direction": t.predicted_direction,
                        "actual_direction": t.actual_direction,
                        "option_symbol": t.option_symbol,
                        "option_type": t.option_type,
                        "strike": t.strike,
                        "expiry": t.expiry,
                        "patterns_count": t.patterns_count,
                        "patterns": t.patterns_str,
                        "spot_price": t.spot_price,
                        "rally_start_utc": t.rally_start_utc.isoformat() if t.rally_start_utc else None,
                        "signal_time_utc": t.signal_time_utc.isoformat() if t.signal_time_utc else None,
                        "planned_entry_time_utc": t.planned_entry_time_utc.isoformat()
                        if t.planned_entry_time_utc
                        else None,
                        "entry_time_utc": t.entry_time_utc.isoformat() if t.entry_time_utc else None,
                        "exit_time_utc": t.exit_time_utc.isoformat() if t.exit_time_utc else None,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "exit_reason": t.exit_reason,
                        "target_hit": 0,
                        "pnl_points": t.pnl_points,
                        "gross_pnl": t.gross_pnl,
                        "transaction_cost": t.transaction_cost,
                        "net_pnl": t.net_pnl,
                        "hold_time_minutes": t.hold_time_minutes,
                    }
                )
            except Exception:
                pass

        self._emit(
            f"MANUAL EXIT | id={t.order_id} | {t.option_symbol} | {entry:.2f}->{t.exit_price:.2f} ({points:.2f} pts) | net={net:.2f} | hold={hold_min:.2f} min | reason={reason} | time={to_ist_str(now_utc)} IST"
        )

    # ---------- exports ----------
    def export_trades_csv(self, out_file: str) -> str:
        with self._lock:
            trades = list(self._completed)
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        with open(out_file, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(
                fp,
                fieldnames=[
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
                    "actual_direction",
                    "predicted_direction",
                    "rally_start",
                    "spot_price",
                    "strike",
                    "option_type",
                    "option_symbol",
                    "expiry",
                    "patterns_count",
                    "patterns",
                    "exit_reason",
                ],
            )
            w.writeheader()
            for t in trades:
                w.writerow(
                    {
                        "Order ID": t.order_id,
                        "Entry Date": to_ist_str(t.entry_time_utc, "%Y-%m-%d"),
                        "Entry Time": to_ist_str(t.entry_time_utc, "%H:%M:%S"),
                        "Enter Price": f"{(t.entry_price or 0):.4f}",
                        "Exit Date": to_ist_str(t.exit_time_utc, "%Y-%m-%d"),
                        "Exit Time": to_ist_str(t.exit_time_utc, "%H:%M:%S"),
                        "Exit Price": f"{(t.exit_price or 0):.4f}",
                        "Pnl Points": f"{(t.pnl_points or 0):.4f}",
                        "gross_pnl": f"{(t.gross_pnl or 0):.2f}",
                        "transaction_cost": f"{(t.transaction_cost or 0):.2f}",
                        "net_pnl": f"{(t.net_pnl or 0):.2f}",
                        "hold_time_minutes": f"{(t.hold_time_minutes or 0):.2f}",
                        "target_hit": str(bool(t.target_hit)),
                        "actual_direction": t.actual_direction,
                        "predicted_direction": t.predicted_direction,
                        "rally_start": to_ist_str(t.rally_start_utc, "%Y-%m-%d %H:%M:%S"),
                        "spot_price": f"{t.spot_price:.2f}",
                        "strike": str(t.strike),
                        "option_type": t.option_type,
                        "option_symbol": t.option_symbol,
                        "expiry": t.expiry,
                        "patterns_count": str(t.patterns_count),
                        "patterns": t.patterns_str,
                        "exit_reason": t.exit_reason or "",
                    }
                )
        return out_file

    def export_trades_xlsx(self, out_file: str) -> str:
        with self._lock:
            trades = list(self._completed)
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        rows = []
        for t in trades:
            rows.append(
                {
                    "Order ID": t.order_id,
                    "Entry Date": to_ist_str(t.entry_time_utc, "%Y-%m-%d"),
                    "Entry Time": to_ist_str(t.entry_time_utc, "%H:%M:%S"),
                    "Enter Price": t.entry_price,
                    "Exit Date": to_ist_str(t.exit_time_utc, "%Y-%m-%d"),
                    "Exit Time": to_ist_str(t.exit_time_utc, "%H:%M:%S"),
                    "Exit Price": t.exit_price,
                    "Pnl Points": t.pnl_points,
                    "gross_pnl": t.gross_pnl,
                    "transaction_cost": t.transaction_cost,
                    "net_pnl": t.net_pnl,
                    "hold_time_minutes": t.hold_time_minutes,
                    "target_hit": t.target_hit,
                    "actual_direction": t.actual_direction,
                    "predicted_direction": t.predicted_direction,
                    "rally_start": to_ist_str(t.rally_start_utc, "%Y-%m-%d %H:%M:%S"),
                    "spot_price": t.spot_price,
                    "strike": t.strike,
                    "option_type": t.option_type,
                    "option_symbol": t.option_symbol,
                    "expiry": t.expiry,
                    "patterns_count": t.patterns_count,
                    "patterns": t.patterns_str,
                    "exit_reason": t.exit_reason,
                }
            )
        df = pd.DataFrame(rows)
        df.to_excel(out_file, index=False, engine="openpyxl")
        return out_file

    def export_db_trades_csv(self, out_file: str, limit: int = 1000) -> Optional[str]:
        if not self._store:
            return None
        rows = self._store.list_trades(limit=limit)
        if not rows:
            return None
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        # write raw DB rows
        cols = list(rows[0].keys())
        with open(out_file, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return out_file

    # ---------- core tick processing ----------
    def on_tick(self, tick: Dict[str, Any], all_symbols: List[str]) -> None:
        """
        Feed a single tick dict.
        `all_symbols` is used to infer nearest expiry.
        """
        ts_utc = parse_ts_utc(tick.get("ts")) or datetime.now(timezone.utc)
        sym = str(tick.get("symbol") or "")
        if not sym:
            return

        with self._lock:
            self._last_feed_ts_utc = ts_utc
            self._latest_tick_by_symbol[sym] = tick

            # Rolling store for option ticks
            if sym.endswith("CE") or sym.endswith("PE"):
                self._option_ticks.append(
                    {
                        "ts": ts_utc,
                        "symbol": sym,
                        "ltp": tick.get("ltp"),
                        "bid": tick.get("bid"),
                        "ask": tick.get("ask"),
                        "volume": tick.get("volume"),
                        "iv": tick.get("iv"),
                        "delta": tick.get("delta"),
                    }
                )
                cutoff = ts_utc - timedelta(seconds=max(120, self.cfg.pattern_window_seconds * 2))
                self._option_ticks = [x for x in self._option_ticks if x["ts"] >= cutoff]

            # Spot candle build + signal
            if sym == "NIFTY 50":
                sp = safe_float(tick.get("ltp"))
                if sp:
                    self._last_spot = sp
                    candle = self._candle_builder.update(ts_utc, float(sp))
                    if candle:
                        self._candles_closed += 1
                        sig = self._detector.on_candle(candle)
                        if sig:
                            actual = "BULLISH" if sig["trend_direction"] == "UP" else "BEARISH"
                            self._on_signal(
                                trend_number=int(sig["trend_number"]),
                                signal_time_utc=sig["signal_time"],
                                spot_price=float(sig["spot_price"]),
                                actual_direction=actual,
                                all_symbols=all_symbols,
                            )

            # Execute pending trade if due
            if self._pending_trade and ts_utc >= self._pending_trade.planned_entry_time_utc:
                if sym == self._pending_trade.option_symbol:
                    entry = fill_price(tick, "BUY", self.cfg.use_bid_ask, self.cfg.slippage_points)
                    if entry:
                        self._pending_trade.entry_time_utc = ts_utc
                        self._pending_trade.entry_price = entry
                        self._active_trade = self._pending_trade
                        self._pending_trade = None
                        self._emit(
                            f"LIVE ENTRY | id={self._active_trade.order_id} trend={self._active_trade.trend_number} {self._active_trade.predicted_direction} | {self._active_trade.option_symbol} @ {entry:.2f} | time={to_ist_str(ts_utc)} IST"
                        )

            # Manage active trade
            if self._active_trade and sym == self._active_trade.option_symbol and self._active_trade.entry_price:
                cur = fill_price(tick, "SELL", self.cfg.use_bid_ask, self.cfg.slippage_points)
                if cur is None:
                    return
                self._maybe_exit_trade(ts_utc, float(cur))

    # ---------- internals ----------
    def _emit(self, line: str) -> None:
        if self._log_fn:
            try:
                self._log_fn(line)
            except Exception:
                pass

    def _format_patterns(self, patterns_detected: Dict[str, float]) -> str:
        if not patterns_detected:
            return "[]"
        items = sorted(patterns_detected.items(), key=lambda kv: kv[0])
        return "[" + ", ".join([f"{k}={v:.3f}" for k, v in items]) + "]"

    def _on_signal(
        self,
        *,
        trend_number: int,
        signal_time_utc: datetime,
        spot_price: float,
        actual_direction: str,
        all_symbols: List[str],
    ) -> None:
        # Single trade at a time
        if self._active_trade or self._pending_trade:
            self._emit(f"Signal {trend_number} ignored: trade already open (active={bool(self._active_trade)} pending={bool(self._pending_trade)})")
            return

        start = signal_time_utc - timedelta(seconds=self.cfg.pattern_window_seconds)
        rows = [x for x in self._option_ticks if start <= x["ts"] < signal_time_utc]
        if not rows:
            self._emit(
                f"Signal {trend_number} skipped: no option ticks in last {self.cfg.pattern_window_seconds}s (warm-up not ready)"
            )
            return

        df = pd.DataFrame(rows)
        for col in ["ltp", "bid", "ask", "volume", "iv", "delta"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["symbol", "ltp"])
        if df.empty:
            self._emit(f"Signal {trend_number} skipped: invalid option window")
            return

        metrics = calculate_fast_indicators(df)
        patterns = detect_fast_patterns(metrics, self.cfg.thresholds)
        if not patterns:
            self._emit(f"Signal {trend_number} skipped: no patterns detected")
            return

        patterns_detected = {p["pattern_type"]: float(p["value"]) for p in patterns if "pattern_type" in p}
        predicted = determine_direction_from_patterns(patterns_detected)
        patterns_str = self._format_patterns(patterns_detected)

        expiry = pick_nearest_expiry(all_symbols, now_utc=signal_time_utc)
        if not expiry:
            self._emit(f"Signal {trend_number} skipped: cannot determine nearest expiry from feed")
            return

        strike = get_atm_strike(float(spot_price))
        option_type = "CE" if predicted == "BULLISH" else "PE"
        option_symbol = f"NIFTY{expiry}{strike}{option_type}"

        self._order_id += 1
        planned_entry = signal_time_utc + timedelta(seconds=self.cfg.latency_seconds)
        self._pending_trade = PaperTrade(
            order_id=self._order_id,
            trend_number=trend_number,
            predicted_direction=predicted,
            actual_direction=actual_direction,
            option_symbol=option_symbol,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            patterns_count=len(patterns_detected),
            patterns_str=patterns_str,
            spot_price=float(spot_price),
            rally_start_utc=signal_time_utc,
            signal_time_utc=signal_time_utc,
            planned_entry_time_utc=planned_entry,
        )

        self._emit(
            f"LIVE SIGNAL | trend={trend_number} {predicted} | spot={spot_price:.2f} | expiry={expiry} | symbol={option_symbol} | patterns={patterns_str} | entry_at={to_ist_str(planned_entry)} IST"
        )

    def _maybe_exit_trade(self, now_utc: datetime, cur_exit: float) -> None:
        t = self._active_trade
        if not t or t.entry_price is None or t.entry_time_utc is None:
            return

        entry = float(t.entry_price)
        target = entry * (1.0 + self.cfg.target_pct / 100.0)
        stop = entry * (1.0 - self.cfg.stop_pct / 100.0)
        hold_min = (now_utc - t.entry_time_utc).total_seconds() / 60.0

        reason: Optional[str] = None
        target_hit = False
        if cur_exit >= target:
            reason = "TARGET"
            target_hit = True
        elif cur_exit <= stop:
            reason = "STOP_LOSS"
        elif hold_min >= self.cfg.max_hold_minutes:
            reason = "TIME_EXIT"

        if not reason:
            return

        t.exit_time_utc = now_utc
        t.exit_price = float(cur_exit)
        t.exit_reason = reason
        t.target_hit = bool(target_hit)

        points = float(t.exit_price) - entry
        gross = points * float(self.cfg.lot_size)
        buy_val = entry * float(self.cfg.lot_size)
        sell_val = float(t.exit_price) * float(self.cfg.lot_size)
        tc = calculate_transaction_cost(buy_val, sell_val)
        net = gross - tc

        t.pnl_points = points
        t.gross_pnl = gross
        t.transaction_cost = tc
        t.net_pnl = net
        t.hold_time_minutes = hold_min

        self._completed.append(t)
        self._active_trade = None

        # Persist completed trade to SQLite
        if self._store:
            try:
                self._store.insert_trade(
                    {
                        "order_id": t.order_id,
                        "trend_number": t.trend_number,
                        "predicted_direction": t.predicted_direction,
                        "actual_direction": t.actual_direction,
                        "option_symbol": t.option_symbol,
                        "option_type": t.option_type,
                        "strike": t.strike,
                        "expiry": t.expiry,
                        "patterns_count": t.patterns_count,
                        "patterns": t.patterns_str,
                        "spot_price": t.spot_price,
                        "rally_start_utc": t.rally_start_utc.isoformat() if t.rally_start_utc else None,
                        "signal_time_utc": t.signal_time_utc.isoformat() if t.signal_time_utc else None,
                        "planned_entry_time_utc": t.planned_entry_time_utc.isoformat()
                        if t.planned_entry_time_utc
                        else None,
                        "entry_time_utc": t.entry_time_utc.isoformat() if t.entry_time_utc else None,
                        "exit_time_utc": t.exit_time_utc.isoformat() if t.exit_time_utc else None,
                        "entry_price": t.entry_price,
                        "exit_price": t.exit_price,
                        "exit_reason": t.exit_reason,
                        "target_hit": 1 if t.target_hit else 0,
                        "pnl_points": t.pnl_points,
                        "gross_pnl": t.gross_pnl,
                        "transaction_cost": t.transaction_cost,
                        "net_pnl": t.net_pnl,
                        "hold_time_minutes": t.hold_time_minutes,
                    }
                )
            except Exception:
                # DB issues should never crash live engine
                pass

        self._emit(
            f"LIVE EXIT | id={t.order_id} | {t.option_symbol} | {entry:.2f}->{t.exit_price:.2f} ({points:.2f} pts) | net={net:.2f} | hold={hold_min:.2f} min | reason={reason} | time={to_ist_str(now_utc)} IST"
        )


