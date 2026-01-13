"""
Pre-Rally Pattern Detection (legacy-style) - BUY-only options.

This ports the proven legacy strategy that produced ~INR 148k net P&L with:
- 30s candles
- trend signal at 0.11% cumulative move
- pattern window 60s (NO overlap with signal candle)
- asymmetric exits: TP=8%, SL=5%, max hold 3 minutes

Implementation notes:
- Uses `snapshot.options_ticks` (session option ticks loaded by SQLiteDataProvider) to compute a rolling
  window of option ticks similar to the legacy engine.
- No lookahead: pattern window is strictly [signal_candle_start - window, signal_candle_start).
- Entry timing: intent emitted at bar end (snapshot.timestamp), matching "signal candle end".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Strategy, Intent
from .registry import register_strategy
from ..data.models import MarketSnapshot


THRESHOLDS_DEFAULT: Dict[str, float] = {
    "iv_change_pct": 5.0,
    "volume_ratio_change": 10.0,
    "delta_change": 0.03,
    "premium_momentum": 2.0,
}


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def calculate_fast_indicators(options_df: pd.DataFrame) -> Dict[str, float]:
    """
    Ported from legacy live_trading_engine.calculate_fast_indicators (simplified; no logging).
    Expects columns: symbol, ltp, volume, iv, delta.
    """
    if options_df is None or options_df.empty:
        return {}

    df = options_df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["option_type"] = df["symbol"].apply(lambda x: "CALL" if x.endswith("CE") else ("PUT" if x.endswith("PE") else "OTHER"))
    calls = df[df["option_type"] == "CALL"].copy()
    puts = df[df["option_type"] == "PUT"].copy()

    metrics: Dict[str, float] = {}

    # IV change %
    if not calls.empty and "iv" in calls.columns:
        g = calls.groupby("symbol", sort=False)
        first = pd.to_numeric(g.first()["iv"], errors="coerce").mean()
        last = pd.to_numeric(g.last()["iv"], errors="coerce").mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            metrics["call_iv_change_pct"] = ((float(last) - float(first)) / float(first)) * 100.0

    if not puts.empty and "iv" in puts.columns:
        g = puts.groupby("symbol", sort=False)
        first = pd.to_numeric(g.first()["iv"], errors="coerce").mean()
        last = pd.to_numeric(g.last()["iv"], errors="coerce").mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            metrics["put_iv_change_pct"] = ((float(last) - float(first)) / float(first)) * 100.0

    # Put/Call volume ratio change (uses first/last puts vs first calls volume baseline)
    if not calls.empty and not puts.empty and "volume" in df.columns:
        cg = calls.groupby("symbol", sort=False)
        pg = puts.groupby("symbol", sort=False)
        call_vol_first = float(pd.to_numeric(cg.first()["volume"], errors="coerce").fillna(0.0).sum())
        put_vol_first = float(pd.to_numeric(pg.first()["volume"], errors="coerce").fillna(0.0).sum())
        put_vol_last = float(pd.to_numeric(pg.last()["volume"], errors="coerce").fillna(0.0).sum())
        if call_vol_first > 0 and put_vol_first > 0:
            pc_first = put_vol_first / call_vol_first
            pc_last = put_vol_last / call_vol_first
            if pc_first > 0:
                metrics["pc_volume_ratio_change"] = ((pc_last - pc_first) / pc_first) * 100.0

    # Delta change
    if not calls.empty and "delta" in calls.columns:
        g = calls.groupby("symbol", sort=False)
        first = pd.to_numeric(g.first()["delta"], errors="coerce").mean()
        last = pd.to_numeric(g.last()["delta"], errors="coerce").mean()
        if pd.notna(first) and pd.notna(last):
            metrics["call_delta_change"] = float(last) - float(first)

    if not puts.empty and "delta" in puts.columns:
        g = puts.groupby("symbol", sort=False)
        first = pd.to_numeric(g.first()["delta"], errors="coerce").mean()
        last = pd.to_numeric(g.last()["delta"], errors="coerce").mean()
        if pd.notna(first) and pd.notna(last):
            metrics["put_delta_change"] = float(last) - float(first)

    # Premium momentum
    if not calls.empty and "ltp" in calls.columns:
        g = calls.groupby("symbol", sort=False)
        first = pd.to_numeric(g.first()["ltp"], errors="coerce").mean()
        last = pd.to_numeric(g.last()["ltp"], errors="coerce").mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            metrics["call_premium_momentum"] = ((float(last) - float(first)) / float(first)) * 100.0

    if not puts.empty and "ltp" in puts.columns:
        g = puts.groupby("symbol", sort=False)
        first = pd.to_numeric(g.first()["ltp"], errors="coerce").mean()
        last = pd.to_numeric(g.last()["ltp"], errors="coerce").mean()
        if pd.notna(first) and pd.notna(last) and float(first) > 0:
            metrics["put_premium_momentum"] = ((float(last) - float(first)) / float(first)) * 100.0

    return metrics


def detect_fast_patterns(metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[Dict[str, Any]]:
    """Ported from legacy detect_fast_patterns (simplified; same semantics)."""
    patterns: List[Dict[str, Any]] = []

    # IV patterns
    if "call_iv_change_pct" in metrics:
        v = float(metrics["call_iv_change_pct"])
        if abs(v) > float(thresholds["iv_change_pct"]):
            patterns.append({"pattern_type": "CALL_IV_CHANGE", "value": v, "direction": "BEARISH" if v > 0 else "BULLISH"})

    if "put_iv_change_pct" in metrics:
        v = float(metrics["put_iv_change_pct"])
        if abs(v) > float(thresholds["iv_change_pct"]):
            patterns.append({"pattern_type": "PUT_IV_CHANGE", "value": v, "direction": "BULLISH" if v < 0 else "BEARISH"})

    # Volume ratio
    if "pc_volume_ratio_change" in metrics:
        v = float(metrics["pc_volume_ratio_change"])
        if abs(v) > float(thresholds["volume_ratio_change"]):
            patterns.append({"pattern_type": "PC_VOLUME_RATIO_CHANGE", "value": v, "direction": "BULLISH" if v < 0 else "BEARISH"})

    # Delta
    if "call_delta_change" in metrics:
        v = float(metrics["call_delta_change"])
        if abs(v) > float(thresholds["delta_change"]):
            patterns.append({"pattern_type": "CALL_DELTA_CHANGE", "value": v, "direction": "BULLISH" if v > 0 else "BEARISH"})

    if "put_delta_change" in metrics:
        v = float(metrics["put_delta_change"])
        if abs(v) > float(thresholds["delta_change"]):
            patterns.append({"pattern_type": "PUT_DELTA_CHANGE", "value": v, "direction": "BEARISH" if v > 0 else "BULLISH"})

    # Premium momentum
    if "call_premium_momentum" in metrics:
        v = float(metrics["call_premium_momentum"])
        if abs(v) > float(thresholds["premium_momentum"]):
            patterns.append({"pattern_type": "CALL_PREMIUM_MOMENTUM", "value": v, "direction": "BULLISH" if v > 0 else "BEARISH"})

    if "put_premium_momentum" in metrics:
        v = float(metrics["put_premium_momentum"])
        if abs(v) > float(thresholds["premium_momentum"]):
            patterns.append({"pattern_type": "PUT_PREMIUM_MOMENTUM", "value": v, "direction": "BEARISH" if v > 0 else "BULLISH"})

    return patterns


def determine_direction_from_patterns(patterns_detected: Dict[str, float]) -> str:
    """Ported from legacy determine_direction_from_patterns()."""
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


@register_strategy("pre_rally_pattern")
class PreRallyPatternStrategy(Strategy):
    """
    Standardized port of the legacy pre-rally pattern strategy.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        self.bar_size = str(params.get("bar_size", "30s"))  # used only for bucket-start calculation
        self.movement_threshold_pct = float(params.get("movement_threshold_pct", 0.11))
        self.pattern_window_seconds = int(params.get("pattern_window_seconds", 60))

        # Filters
        self.require_patterns_count: Optional[int] = params.get("require_patterns_count", 1)
        self.skip_conflicts: bool = bool(params.get("skip_conflicts", False))
        self.allow_pattern_types: Optional[List[str]] = params.get("allow_pattern_types", None)
        if self.allow_pattern_types:
            self.allow_pattern_types = [str(x).strip() for x in self.allow_pattern_types if str(x).strip()]

        # Exits (percent values)
        self.take_profit_pct = float(params.get("take_profit_pct", 8.0))
        self.stop_loss_pct = float(params.get("stop_loss_pct", 5.0))
        self.max_hold_minutes = float(params.get("max_hold_minutes", 3.0))

        # Sizing
        self.contracts = int(params.get("contracts", 1))

        # Thresholds
        self.thresholds = dict(THRESHOLDS_DEFAULT)
        self.thresholds.update(params.get("thresholds", {}) or {})

        # Trend detector state
        self._current_direction: Optional[int] = None
        self._direction_start_price: Optional[float] = None
        self._already_signaled: bool = False

    def _spot_direction(self, o: float, c: float) -> int:
        if o <= 0:
            return 0
        pct_change = ((c - o) / o) * 100.0
        if pct_change > 0.01:
            return 1
        if pct_change < -0.01:
            return -1
        return 0

    def _bar_seconds(self) -> float:
        try:
            return float(pd.to_timedelta(self.bar_size).total_seconds())
        except Exception:
            return 30.0

    def _extract_option_window(self, snapshot: MarketSnapshot, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
        df = snapshot.options_ticks
        if df is None or df.empty:
            return pd.DataFrame()
        out = df
        if "ts" in out.columns:
            ts = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        else:
            return pd.DataFrame()
        mask = (ts >= window_start) & (ts < window_end)
        out = out.loc[mask].copy()
        if out.empty:
            return out
        # normalize to legacy column names for indicator code
        out = out.rename(columns={"ltp": "ltp"})
        out["symbol"] = out["symbol"].astype(str)
        # keep only CE/PE options (defensive)
        out = out[(out["symbol"].str.endswith("CE") | out["symbol"].str.endswith("PE"))]
        # require ltp > 0
        out["ltp"] = pd.to_numeric(out.get("ltp"), errors="coerce")
        out = out[out["ltp"].fillna(0) > 0]
        return out

    def generate_intents(self, snapshot: MarketSnapshot, context: Optional[Dict[str, Any]] = None) -> List[Intent]:
        if not snapshot.has_spot():
            return []
        sb = snapshot.spot_bar.iloc[0]
        o = _safe_float(sb.get("open"))
        c = _safe_float(sb.get("close"))
        if o is None or c is None:
            return []

        d = self._spot_direction(o, c)

        # Reset trend tracking on direction change or neutral candle
        if d != self._current_direction or d == 0:
            self._current_direction = d
            self._direction_start_price = o
            self._already_signaled = False
            return []

        # Skip if already signaled this trend leg
        if self._already_signaled or self._direction_start_price is None:
            return []

        cumulative_pct = ((c - float(self._direction_start_price)) / float(self._direction_start_price)) * 100.0 if float(self._direction_start_price) != 0 else 0.0
        if abs(cumulative_pct) < self.movement_threshold_pct:
            return []

        # Mark signaled (whether or not we find patterns); legacy does not retry within same leg.
        self._already_signaled = True

        # Pattern window ends at signal candle START (no overlap with the signal candle).
        bar_seconds = self._bar_seconds()
        signal_candle_start = pd.to_datetime(snapshot.timestamp, utc=True) - pd.Timedelta(seconds=bar_seconds)
        window_end = signal_candle_start
        window_start = window_end - pd.Timedelta(seconds=int(self.pattern_window_seconds))

        opt_window = self._extract_option_window(snapshot, window_start=window_start, window_end=window_end)
        if opt_window.empty:
            return []

        metrics = calculate_fast_indicators(opt_window)
        if not metrics:
            return []
        patterns = detect_fast_patterns(metrics, self.thresholds)
        if not patterns:
            return []

        # Filters
        if self.allow_pattern_types:
            allowed = set(self.allow_pattern_types)
            patterns = [p for p in patterns if str(p.get("pattern_type", "")).strip() in allowed]
            if not patterns:
                return []

        if self.require_patterns_count is not None and len(patterns) != int(self.require_patterns_count):
            return []

        if self.skip_conflicts:
            dirs = {str(p.get("direction", "")).strip().upper() for p in patterns}
            if "BULLISH" in dirs and "BEARISH" in dirs:
                return []

        patterns_dict = {str(p["pattern_type"]): float(p["value"]) for p in patterns if "pattern_type" in p and "value" in p}
        pred = determine_direction_from_patterns(patterns_dict)
        option_type = "CALL" if pred == "BULLISH" else "PUT"

        max_hold_bars = int(np.ceil((self.max_hold_minutes * 60.0) / max(bar_seconds, 1.0)))

        meta = {
            "reason": "pre_rally_pattern",
            "movement_threshold_pct": self.movement_threshold_pct,
            "cumulative_move_pct": float(cumulative_pct),
            "pattern_window_seconds": int(self.pattern_window_seconds),
            "patterns": patterns,
            "metrics": metrics,
            # exits
            "stop_loss_pct": float(self.stop_loss_pct),
            "take_profit_pct": float(self.take_profit_pct),
            "max_hold_bars": int(max_hold_bars),
        }

        return [
            Intent(
                timestamp=snapshot.timestamp,
                direction="LONG",
                option_type=option_type,  # type: ignore[arg-type]
                size=int(max(1, self.contracts)),
                metadata=meta,
            )
        ]


