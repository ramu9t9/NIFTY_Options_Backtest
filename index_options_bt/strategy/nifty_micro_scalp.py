"""
NIFTY Micro-Scalp Strategy (BUY-only).

Goal: capture continuation for the next 1–2 candles after an impulse move.

Design goals:
- Uses SPOT + FUT bars + option chain greeks (delta/iv/oi/volume) from SQLite snapshots.
- Does NOT rely on bid/ask (DB does not provide usable values).
- Avoid lookahead: signal is detected on bar close, entry happens on the NEXT bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np
import pandas as pd

from .base import Strategy, Intent
from .registry import register_strategy
from ..data.models import MarketSnapshot


Direction = Literal["UP", "DOWN"]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _ema(prev: Optional[float], x: float, period: int) -> float:
    k = 2.0 / (float(period) + 1.0)
    if prev is None:
        return float(x)
    return float(x) * k + float(prev) * (1.0 - k)


def _zscore(x: float, xs: List[float]) -> float:
    if len(xs) < 5:
        return 0.0
    arr = np.asarray(xs, dtype=float)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if sd <= 1e-12:
        return 0.0
    return (float(x) - mu) / sd


@register_strategy("nifty_micro_scalp")
class NiftyMicroScalpStrategy(Strategy):
    """
    BUY-only micro-scalp strategy for NIFTY options.

    Params (all optional; sensible defaults provided):
      - range_z_lookback: int (default 40)
      - range_z_threshold: float (default 1.4)
      - body_ratio_threshold: float (default 0.65)
      - close_pos_threshold: float (default 0.85)
      - fut_ema_fast: int (default 9)
      - fut_ema_slow: int (default 21)
      - fut_breakout_lookback: int (default 3)
      - fut_momentum_threshold_pct: float (default 0.02)  # optional, in %
      - use_fut_breakout: bool (default True)
      - delta_min: float (default 0.55)
      - delta_max: float (default 0.75)
      - iv_z_lookback: int (default 40)
      - iv_z_max: float (default 1.0)  # avoid buying IV top
      - vol_rise_lookback: int (default 6)
      - vol_rise_ratio: float (default 1.2)
      - oi_drop_max_ratio: float (default 0.15)  # 0.15 means allow max 15% drop
      - require_oi_non_collapse: bool (default False)
      - chop_flip_lookback: int (default 10)
      - chop_flip_max: int (default 6)
      - max_trades_per_day: int (default 3)
      - cooldown_bars: int (default 4)

    Exits (passed via intent metadata to portfolio/runner):
      - stop_loss_pct: float (default 18.0)
      - take_profit_pct_1: Optional[float] (default 12.0)
      - take_profit_pct_2: float (default 25.0)
      - tp1_fraction: float (default 0.5)
      - max_hold_bars: int (default 2)
      - time_stop_bars: int (default 2)
      - min_profit_pct_by_time_stop: float (default 0.0)
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Stage A (spot impulse)
        self.range_z_lookback = int(params.get("range_z_lookback", 40))
        self.range_z_threshold = float(params.get("range_z_threshold", 1.4))
        self.body_ratio_threshold = float(params.get("body_ratio_threshold", 0.65))
        self.close_pos_threshold = float(params.get("close_pos_threshold", 0.85))

        # Stage B (futures confirm)
        self.fut_ema_fast = int(params.get("fut_ema_fast", 9))
        self.fut_ema_slow = int(params.get("fut_ema_slow", 21))
        self.fut_breakout_lookback = int(params.get("fut_breakout_lookback", 3))
        self.fut_momentum_threshold_pct = float(params.get("fut_momentum_threshold_pct", 0.02))
        self.use_fut_breakout = bool(params.get("use_fut_breakout", True))

        # Stage C (options confirm)
        self.delta_min = float(params.get("delta_min", 0.55))
        self.delta_max = float(params.get("delta_max", 0.75))
        self.iv_z_lookback = int(params.get("iv_z_lookback", 40))
        self.iv_z_max = float(params.get("iv_z_max", 1.0))
        self.vol_rise_lookback = int(params.get("vol_rise_lookback", 6))
        self.vol_rise_ratio = float(params.get("vol_rise_ratio", 1.2))
        self.require_oi_non_collapse = bool(params.get("require_oi_non_collapse", False))
        self.oi_drop_max_ratio = float(params.get("oi_drop_max_ratio", 0.15))

        # Chop + trade limits
        self.chop_flip_lookback = int(params.get("chop_flip_lookback", 10))
        self.chop_flip_max = int(params.get("chop_flip_max", 6))
        self.max_trades_per_day = int(params.get("max_trades_per_day", 3))
        self.cooldown_bars = int(params.get("cooldown_bars", 4))

        # Exits
        self.stop_loss_pct = float(params.get("stop_loss_pct", 18.0))
        self.take_profit_pct_1 = params.get("take_profit_pct_1", 12.0)
        self.take_profit_pct_2 = float(params.get("take_profit_pct_2", 25.0))
        self.tp1_fraction = float(params.get("tp1_fraction", 0.5))
        self.max_hold_bars = int(params.get("max_hold_bars", 2))
        self.time_stop_bars = int(params.get("time_stop_bars", 2))
        self.min_profit_pct_by_time_stop = float(params.get("min_profit_pct_by_time_stop", 0.0))

        # State (spot)
        self._spot_ranges: List[float] = []
        self._spot_rets_sign: List[int] = []

        # State (futures)
        self._fut_close_hist: List[float] = []
        self._fut_ema_fast: Optional[float] = None
        self._fut_ema_slow: Optional[float] = None

        # State (options by (cp,target_strike,expiry))
        self._opt_iv_hist: Dict[str, List[float]] = {}
        self._opt_vol_hist: Dict[str, List[float]] = {}
        self._opt_oi_hist: Dict[str, List[float]] = {}

        # Entry scheduling (avoid lookahead)
        self._pending: Optional[Tuple[Direction, Dict[str, Any]]] = None

        # Trade pacing
        self._last_entry_bar_idx: Optional[int] = None
        self._bar_idx: int = 0
        self._trades_by_day: Dict[str, int] = {}

    def _spot_impulse(self, snapshot: MarketSnapshot) -> Tuple[Optional[Direction], Dict[str, Any]]:
        sb = snapshot.spot_bar.iloc[0]
        o = _safe_float(sb.get("open"))
        h = _safe_float(sb.get("high"))
        l = _safe_float(sb.get("low"))
        c = _safe_float(sb.get("close"))
        if o is None or h is None or l is None or c is None:
            return None, {"reason": "spot_missing_ohlc"}
        rng = float(h - l)
        if rng <= 0:
            return None, {"reason": "spot_zero_range"}

        body = abs(c - o)
        body_ratio = body / rng
        close_pos_up = (c - l) / rng
        close_pos_dn = (h - c) / rng

        # update histories
        self._spot_ranges.append(rng)
        if len(self._spot_ranges) > max(self.range_z_lookback, 10):
            self._spot_ranges = self._spot_ranges[-max(self.range_z_lookback, 10) :]

        # return sign for chop filter
        if len(self._spot_rets_sign) > 0:
            pass
        # approximate sign from close-open
        ret_sign = 1 if c > o else (-1 if c < o else 0)
        self._spot_rets_sign.append(ret_sign)
        if len(self._spot_rets_sign) > max(self.chop_flip_lookback, 10):
            self._spot_rets_sign = self._spot_rets_sign[-max(self.chop_flip_lookback, 10) :]

        rng_z = _zscore(rng, self._spot_ranges[-self.range_z_lookback :])
        meta = {
            "spot_open": o,
            "spot_high": h,
            "spot_low": l,
            "spot_close": c,
            "range": rng,
            "range_z": rng_z,
            "body_ratio": body_ratio,
            "close_pos_up": close_pos_up,
            "close_pos_dn": close_pos_dn,
        }

        if rng_z < self.range_z_threshold:
            return None, {**meta, "reason": "range_z_below"}
        if body_ratio < self.body_ratio_threshold:
            return None, {**meta, "reason": "body_ratio_below"}

        if c > o and close_pos_up >= self.close_pos_threshold:
            return "UP", {**meta, "reason": "spot_impulse_up"}
        if c < o and close_pos_dn >= self.close_pos_threshold:
            return "DOWN", {**meta, "reason": "spot_impulse_down"}
        return None, {**meta, "reason": "spot_not_extreme"}

    def _fut_confirm(self, snapshot: MarketSnapshot, direction: Direction) -> Tuple[bool, Dict[str, Any]]:
        if snapshot.futures_bar is None or snapshot.futures_bar.empty:
            return False, {"reason": "futures_missing"}
        fb = snapshot.futures_bar.iloc[0]
        c = _safe_float(fb.get("close"))
        h = _safe_float(fb.get("high"))
        l = _safe_float(fb.get("low"))
        if c is None or h is None or l is None:
            return False, {"reason": "futures_bad_ohlc"}

        # State is updated every bar in generate_intents(). Keep this here too for safety.
        self._fut_close_hist.append(float(c))
        if len(self._fut_close_hist) > 200:
            self._fut_close_hist = self._fut_close_hist[-200:]

        self._fut_ema_fast = _ema(self._fut_ema_fast, float(c), self.fut_ema_fast)
        self._fut_ema_slow = _ema(self._fut_ema_slow, float(c), self.fut_ema_slow)

        ema_ok = (self._fut_ema_fast is not None) and (self._fut_ema_slow is not None)
        if not ema_ok:
            return False, {"reason": "futures_ema_warmup"}

        if direction == "UP":
            align = self._fut_ema_fast > self._fut_ema_slow
        else:
            align = self._fut_ema_fast < self._fut_ema_slow
        if not align:
            return False, {"reason": "futures_ema_misaligned", "ema_fast": self._fut_ema_fast, "ema_slow": self._fut_ema_slow}

        if not self.use_fut_breakout:
            return True, {"reason": "futures_confirm_no_breakout", "ema_fast": self._fut_ema_fast, "ema_slow": self._fut_ema_slow}

        # Breakout over last N bars (using close history as proxy)
        lb = max(int(self.fut_breakout_lookback), 1)
        if len(self._fut_close_hist) < lb + 1:
            return False, {"reason": "futures_breakout_warmup"}
        recent = self._fut_close_hist[-(lb + 1) : -1]
        hi = float(np.max(recent))
        lo = float(np.min(recent))

        mom = 0.0
        try:
            prev = float(self._fut_close_hist[-2])
            mom = (float(c) / prev - 1.0) * 100.0 if prev != 0 else 0.0
        except Exception:
            mom = 0.0

        if direction == "UP":
            ok = (float(c) > hi) or (mom >= self.fut_momentum_threshold_pct)
        else:
            ok = (float(c) < lo) or (mom <= -abs(self.fut_momentum_threshold_pct))

        return bool(ok), {
            "reason": "futures_confirm" if ok else "futures_no_breakout",
            "ema_fast": float(self._fut_ema_fast),
            "ema_slow": float(self._fut_ema_slow),
            "break_hi": hi,
            "break_lo": lo,
            "momentum_pct": mom,
        }

    def _update_futures_state(self, snapshot: MarketSnapshot) -> None:
        """Warm up futures EMA state continuously so signals don't depend on a prior impulse."""
        if snapshot.futures_bar is None or snapshot.futures_bar.empty:
            return
        fb = snapshot.futures_bar.iloc[0]
        c = _safe_float(fb.get("close"))
        if c is None:
            return
        self._fut_close_hist.append(float(c))
        if len(self._fut_close_hist) > 200:
            self._fut_close_hist = self._fut_close_hist[-200:]
        self._fut_ema_fast = _ema(self._fut_ema_fast, float(c), self.fut_ema_fast)
        self._fut_ema_slow = _ema(self._fut_ema_slow, float(c), self.fut_ema_slow)

    def _chop_filter(self) -> Tuple[bool, Dict[str, Any]]:
        n = max(int(self.chop_flip_lookback), 2)
        if len(self._spot_rets_sign) < n:
            return False, {"reason": "chop_warmup"}
        xs = self._spot_rets_sign[-n:]
        flips = 0
        prev = xs[0]
        for x in xs[1:]:
            if x == 0 or prev == 0:
                prev = x
                continue
            if x != prev:
                flips += 1
            prev = x
        if flips > int(self.chop_flip_max):
            return True, {"reason": "chop_flip_too_high", "flip_count": flips, "lookback": n}
        return False, {"reason": "chop_ok", "flip_count": flips, "lookback": n}

    def _pick_candidate_option_row(self, snapshot: MarketSnapshot, direction: Direction) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
        if snapshot.options_chain is None or snapshot.options_chain.empty:
            return None, {"reason": "options_chain_missing"}
        spot = snapshot.get_spot_price()
        if spot is None:
            return None, {"reason": "spot_missing_for_option_pick"}

        chain = snapshot.options_chain.copy()
        # nearest expiry
        if "expiry" not in chain.columns:
            return None, {"reason": "chain_missing_expiry"}
        chain = chain[chain["last"].fillna(0) > 0]
        if chain.empty:
            return None, {"reason": "chain_no_ltp"}
        # normalize strike/cp
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        chain = chain.dropna(subset=["strike"])

        step = int(self.params.get("strike_step", 50))
        atm = int(round(float(spot) / step) * step)
        if direction == "UP":
            cp = "C"
            target_strike = atm - step
        else:
            cp = "P"
            target_strike = atm + step

        chain = chain[chain["cp"] == cp]
        chain = chain.dropna(subset=["expiry"])
        # choose nearest expiry
        exp = sorted(chain["expiry"].unique())[0]
        chain = chain[chain["expiry"] == exp]

        chain["abs_strike_target"] = (chain["strike"] - float(target_strike)).abs()
        chain = chain.sort_values(by=["abs_strike_target", "oi", "volume", "symbol"], ascending=[True, False, False, True], kind="mergesort")
        row = chain.iloc[0]
        return row, {
            "picked_cp": cp,
            "picked_expiry": str(exp),
            "picked_target_strike": int(target_strike),
            "picked_strike": int(row.get("strike")),
            "picked_symbol": str(row.get("symbol")),
        }

    def _options_confirm(self, snapshot: MarketSnapshot, direction: Direction) -> Tuple[bool, Dict[str, Any]]:
        row, meta_pick = self._pick_candidate_option_row(snapshot, direction)
        if row is None:
            return False, meta_pick

        sym = str(row.get("symbol"))
        delta = _safe_float(row.get("delta"))
        iv = _safe_float(row.get("iv"))
        vol = _safe_float(row.get("volume"))
        oi = _safe_float(row.get("oi"))

        meta = {**meta_pick, "delta": delta, "iv": iv, "volume": vol, "oi": oi}

        if delta is None or not (self.delta_min <= abs(delta) <= self.delta_max):
            return False, {**meta, "reason": "delta_out_of_band"}

        # IV spike filter
        if iv is not None:
            hist = self._opt_iv_hist.setdefault(sym, [])
            hist.append(float(iv))
            if len(hist) > max(self.iv_z_lookback, 10):
                self._opt_iv_hist[sym] = hist[-max(self.iv_z_lookback, 10) :]
            z = _zscore(float(iv), self._opt_iv_hist[sym][-self.iv_z_lookback :])
            meta["iv_z"] = z
            if z > float(self.iv_z_max):
                return False, {**meta, "reason": "iv_spike"}

        # Volume rising filter
        if vol is not None:
            vh = self._opt_vol_hist.setdefault(sym, [])
            vh.append(float(vol))
            if len(vh) > max(self.vol_rise_lookback, 10):
                self._opt_vol_hist[sym] = vh[-max(self.vol_rise_lookback, 10) :]
            look = self._opt_vol_hist[sym][-self.vol_rise_lookback :]
            base = float(np.mean(look[:-1])) if len(look) >= 2 else float(np.mean(look))
            base = max(base, 1e-9)
            ratio = float(vol) / base
            meta["vol_ratio"] = ratio
            if ratio < float(self.vol_rise_ratio):
                return False, {**meta, "reason": "vol_not_rising"}

        # OI not collapsing (optional)
        if self.require_oi_non_collapse and oi is not None:
            oh = self._opt_oi_hist.setdefault(sym, [])
            oh.append(float(oi))
            if len(oh) > 50:
                self._opt_oi_hist[sym] = oh[-50:]
            if len(oh) >= 2:
                prev = float(oh[-2])
                if prev > 0:
                    drop = max(0.0, (prev - float(oi)) / prev)
                    meta["oi_drop_ratio"] = drop
                    if drop > float(self.oi_drop_max_ratio):
                        return False, {**meta, "reason": "oi_collapse"}

        return True, {**meta, "reason": "options_confirm"}

    def generate_intents(self, snapshot: MarketSnapshot, context: Optional[Dict[str, Any]] = None) -> List[Intent]:
        self._bar_idx += 1
        if not snapshot.has_spot():
            return []

        # Warm up futures indicators regardless of whether a spot impulse triggers this bar.
        self._update_futures_state(snapshot)

        # 1) Emit pending entry on NEXT bar (avoid lookahead)
        if self._pending is not None:
            direction, pending_meta = self._pending
            self._pending = None

            day_key = snapshot.timestamp.date().isoformat()
            self._trades_by_day.setdefault(day_key, 0)
            if self._trades_by_day[day_key] >= self.max_trades_per_day:
                return [Intent(timestamp=snapshot.timestamp, direction="FLAT", metadata={"reason": "max_trades_per_day"})]

            if self._last_entry_bar_idx is not None and (self._bar_idx - self._last_entry_bar_idx) < self.cooldown_bars:
                return []

            self._trades_by_day[day_key] += 1
            self._last_entry_bar_idx = self._bar_idx

            opt_type = "CALL" if direction == "UP" else "PUT"
            meta = {
                **pending_meta,
                "stop_loss_pct": float(self.stop_loss_pct),
                # TP2 is stored in take_profit_pct for compatibility
                "take_profit_pct": float(self.take_profit_pct_2),
                "take_profit_pct_1": float(self.take_profit_pct_1) if self.take_profit_pct_1 is not None else None,
                "take_profit_pct_2": float(self.take_profit_pct_2),
                "tp1_fraction": float(self.tp1_fraction),
                "max_hold_bars": int(self.max_hold_bars),
                "time_stop_bars": int(self.time_stop_bars),
                "min_profit_pct_by_time_stop": float(self.min_profit_pct_by_time_stop),
            }
            return [
                Intent(
                    timestamp=snapshot.timestamp,
                    direction="LONG",
                    option_type=opt_type,  # type: ignore[arg-type]
                    size=1,
                    metadata=meta,
                )
            ]

        # 2) Build signal on this bar close and schedule entry for next bar
        chopped, chop_meta = self._chop_filter()
        if chopped:
            return []

        spot_dir, spot_meta = self._spot_impulse(snapshot)
        if spot_dir is None:
            return []

        fut_ok, fut_meta = self._fut_confirm(snapshot, spot_dir)
        if not fut_ok:
            return []

        opt_ok, opt_meta = self._options_confirm(snapshot, spot_dir)
        if not opt_ok:
            return []

        # schedule entry next bar
        self._pending = (
            spot_dir,
            {
                "signal_ts": snapshot.timestamp,
                "signal_reason": "micro_scalp_signal",
                "spot": spot_meta,
                "futures": fut_meta,
                "options": opt_meta,
                "chop": chop_meta,
            },
        )
        return []



