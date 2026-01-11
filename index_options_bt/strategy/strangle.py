"""
Short Strangle Strategy (multi-leg example).

This is a minimal, production-friendly example of a multi-leg strategy using the existing engine:
- Generates **two intents** on entry: SHORT CALL + SHORT PUT
- Exits by emitting a **FLAT** intent (close all legs) based on time or optional TP/SL (per-leg)

Notes:
- Contract selection is delegated to the selector. Recommended selector mode: "delta"
  with target_delta=0.25 (puts will use -0.25 automatically).
- Database has bid/ask=0; execution uses LTP fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import Strategy, Intent
from .registry import register_strategy
from ..data.models import MarketSnapshot


@register_strategy("strangle")
class ShortStrangleStrategy(Strategy):
    """
    Enter a short strangle once per day at (or after) a configured time.

    Params:
      - entry_time_ist: str, e.g. "10:00"
      - max_hold_bars: int, optional (default 240)  # ~1 hour on 15s bars
      - take_profit_pct: float, optional (applied per leg; for shorts TP triggers on premium decay)
      - stop_loss_pct: float, optional (applied per leg)
      - size: int, contracts per leg (default 1)
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.entry_time_ist = str(params.get("entry_time_ist", "10:00"))
        self.max_hold_bars = params.get("max_hold_bars", 240)
        self.take_profit_pct = params.get("take_profit_pct", None)
        self.stop_loss_pct = params.get("stop_loss_pct", None)
        self.size = int(params.get("size", 1))

        self._active: bool = False
        self._entry_bar_idx: Optional[int] = None
        self._bar_idx: int = 0
        self._last_day_ist: Optional[str] = None

    def _ts_ist(self, ts_utc) -> pd.Timestamp:
        t = pd.Timestamp(ts_utc)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t.tz_convert("Asia/Kolkata")

    def generate_intents(self, snapshot: MarketSnapshot, context: Optional[Dict[str, Any]] = None) -> List[Intent]:
        if not snapshot.has_spot():
            return []

        self._bar_idx += 1

        ts_ist = self._ts_ist(snapshot.timestamp)
        day_key = ts_ist.date().isoformat()

        # Reset state on new day
        if self._last_day_ist != day_key:
            self._last_day_ist = day_key
            self._active = False
            self._entry_bar_idx = None

        # Exit by max_hold_bars
        if self._active and self._entry_bar_idx is not None and self.max_hold_bars is not None:
            if self._bar_idx - self._entry_bar_idx >= int(self.max_hold_bars):
                self._active = False
                self._entry_bar_idx = None
                return [Intent(timestamp=snapshot.timestamp, direction="FLAT", metadata={"reason": "time_exit"})]

        # Entry condition: first bar at/after entry_time_ist (once per day)
        if self._active:
            return []

        try:
            hh, mm = [int(x) for x in self.entry_time_ist.split(":")]
        except Exception:
            hh, mm = 10, 0

        if (ts_ist.hour, ts_ist.minute) < (hh, mm):
            return []

        meta: Dict[str, Any] = {"reason": "scheduled_entry", "entry_time_ist": self.entry_time_ist}
        if self.stop_loss_pct is not None:
            meta["stop_loss_pct"] = self.stop_loss_pct
        if self.take_profit_pct is not None:
            meta["take_profit_pct"] = self.take_profit_pct
        if self.max_hold_bars is not None:
            meta["max_hold_bars"] = int(self.max_hold_bars)

        call_intent = Intent(
            timestamp=snapshot.timestamp,
            direction="SHORT",
            option_type="CALL",
            size=int(self.size),
            metadata=dict(meta),
        )
        put_intent = Intent(
            timestamp=snapshot.timestamp,
            direction="SHORT",
            option_type="PUT",
            size=int(self.size),
            metadata=dict(meta),
        )

        self._active = True
        self._entry_bar_idx = self._bar_idx
        return [call_intent, put_intent]


