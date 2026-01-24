"""
NIFTY Institutional Flow + Momentum Capture (BUY-only | Index Options | Intraday)

Design constraints (repo contract):
- LTP-only is reliable (no bid/ask/mid microstructure).
- OI/volume/IV/greeks may be missing/zero -> use defensively; optionally require them via params.
- Use closed bars only (no lookahead). Entries are next-bar open (implemented via pending schedule).

Strategy (high level):
1) Futures Direction Engine:
   - 15m VWAP computed from rolling base bars (close*volume / volume).
   - 1H EMA20/EMA50 computed from closed 1H futures bars.
   - If direction not clean (bullish or bearish), do not trade.
2) Institutional Confirmation (options):
   - Determine ATM strike from spot close: round(spot/step)*step
   - Consider strikes: ATM, ATM±1 step, ATM±2 steps (default step=50)
   - For candidate side (CE for bullish, PE for bearish), require:
       ΔOI(5m) > 0 and ΔLTP(5m) > 0 (thresholded)
     And opposite side shows weakness (any of the window):
       ΔOI(5m) <= 0 OR ΔLTP(5m) <= 0 (thresholded)
3) Greeks Stability Filter (optional/defensive):
   - Delta band, gamma min, theta acceleration max, IV stability max change.
4) Entry Engine:
   - Futures breaks the HIGH/LOW of the previous closed 3-minute bar (tracked by state).
   - Option premium breaks the HIGH/LOW of the previous closed 1-minute bar for the candidate symbol.
   - "Aggressive buyers" proxy (LTP impulse) replaces bid/ask imbalance.
   - Signal is detected on bar close; entry is scheduled for NEXT bar to avoid lookahead.
5) Exits:
   - Hard TP/SL handled by engine via intent metadata (pct-based).
   - Optional strategy exit when futures re-enters VWAP zone or delta collapses vs entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import logging

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from .base import Strategy, Intent
from .registry import register_strategy
from .context import StrategyContext
from ..data.models import MarketSnapshot


logger = logging.getLogger(__name__)

Direction = Literal["BULLISH", "BEARISH"]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _ema(prev: Optional[float], x: float, period: int) -> float:
    k = 2.0 / (float(period) + 1.0)
    if prev is None:
        return float(x)
    return float(x) * k + float(prev) * (1.0 - k)


@dataclass
class _Sample:
    ts: datetime
    ltp: Optional[float] = None
    oi: Optional[float] = None
    volume: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    iv: Optional[float] = None


class _SeriesTracker:
    """Lightweight as-of tracker for a single option symbol."""

    def __init__(self, keep_minutes: float = 90.0):
        self.keep_td = timedelta(minutes=float(keep_minutes))
        self.samples: List[_Sample] = []

    def update(self, s: _Sample) -> None:
        self.samples.append(s)
        # prune old
        cutoff = s.ts - self.keep_td
        # list is in time order under runner; prune from left
        i = 0
        while i < len(self.samples) and self.samples[i].ts < cutoff:
            i += 1
        if i > 0:
            self.samples = self.samples[i:]

    def asof(self, ts: datetime) -> Optional[_Sample]:
        if not self.samples:
            return None
        # since samples are ordered, walk from end
        for s in reversed(self.samples):
            if s.ts <= ts:
                return s
        return None


@register_strategy("institutional_flow_momentum")
class InstitutionalFlowMomentumStrategy(Strategy):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # --- Time filters (IST) ---
        self.tz = ZoneInfo(str(params.get("timezone", "Asia/Kolkata")))
        self.allowed_sessions: List[Tuple[time, time]] = []
        for w in params.get("allowed_windows_ist", [["09:25", "11:30"], ["13:45", "14:45"]]):
            try:
                a, b = str(w[0]), str(w[1])
                self.allowed_sessions.append((time.fromisoformat(a), time.fromisoformat(b)))
            except Exception:
                continue

        # --- Futures direction engine ---
        self.vwap_window_minutes = float(params.get("vwap_window_minutes", 15.0))
        self.ema_fast_period = int(params.get("ema_fast_period", 20))  # EMA20 on 1H
        self.ema_slow_period = int(params.get("ema_slow_period", 50))  # EMA50 on 1H
        self.vwap_reentry_band_pct = float(params.get("vwap_reentry_band_pct", 0.05))  # 0.05% band

        # --- Option universe ---
        self.strike_step = int(params.get("strike_step", 50))
        self.atm_window_steps = int(params.get("atm_window_steps", 2))  # ATM±2 steps => ±100
        self.nearest_expiry_only = bool(params.get("nearest_expiry_only", True))
        # Keep strategy's candidate selection aligned with selector delta mode.
        self.target_delta = float(params.get("target_delta", 0.55))

        # --- OI/Premium confirmation (5m deltas) ---
        self.delta_window_minutes = float(params.get("delta_window_minutes", 5.0))
        self.min_oi_change_pct = float(params.get("min_oi_change_pct", 1.5))
        self.min_premium_change_pct = float(params.get("min_premium_change_pct", 1.0))
        self.min_volume_change_pct = float(params.get("min_volume_change_pct", 0.0))
        self.require_oi_confirmation = bool(params.get("require_oi_confirmation", True))
        self.require_volume_confirmation = bool(params.get("require_volume_confirmation", False))
        self.opposite_weak_threshold_pct = float(params.get("opposite_weak_threshold_pct", 0.5))

        # --- Greeks stability (optional/defensive) ---
        self.require_greeks = bool(params.get("require_greeks", False))
        self.delta_min = float(params.get("delta_min", 0.45))
        self.delta_max = float(params.get("delta_max", 0.65))
        self.gamma_min = float(params.get("gamma_min", 0.001))
        self.theta_accel_max_abs = float(params.get("theta_accel_max_abs", 0.15))  # abs change over 5m
        self.iv_change_max_pct = float(params.get("iv_change_max_pct", 8.0))

        # --- Entry breakout engine ---
        self.use_aggressive_proxy = bool(params.get("use_aggressive_proxy", True))
        self.min_opt_impulse_pct_5s = float(params.get("min_opt_impulse_pct_5s", 0.25))
        self.min_fut_impulse_pct_5s = float(params.get("min_fut_impulse_pct_5s", 0.01))

        # --- Risk/targets (engine-level exits via metadata) ---
        self.stop_loss_pct = float(params.get("stop_loss_pct", 25.0))
        self.r_multiple_t1 = float(params.get("r_multiple_t1", 1.5))
        self.r_multiple_t2 = float(params.get("r_multiple_t2", 3.0))
        self.tp1_fraction = float(params.get("tp1_fraction", 0.5))
        self.max_hold_minutes = float(params.get("max_hold_minutes", 8.0))
        self.contracts = int(params.get("contracts", 1))

        # --- State ---
        self._trackers: Dict[str, _SeriesTracker] = {}
        self._pending: Optional[Tuple[str, Dict[str, Any]]] = None  # ("CALL"/"PUT", metadata)

        # futures base history for VWAP
        self._fut_vwap_pxv: List[float] = []
        self._fut_vwap_v: List[float] = []
        self._fut_prev_close: Optional[float] = None

        # 1H EMA state (updated on new closed 1h bar)
        self._ema20_1h: Optional[float] = None
        self._ema50_1h: Optional[float] = None
        self._last_1h_ts: Optional[datetime] = None

        # Previous closed bars tracking for breakout (avoid lookahead)
        self._last_3m_ts: Optional[datetime] = None
        self._prev_3m_bar: Optional[Dict[str, float]] = None
        self._last_3m_bar: Optional[Dict[str, float]] = None

        self._last_opt_1m_ts: Dict[str, datetime] = {}
        self._prev_opt_1m_bar: Dict[str, Dict[str, float]] = {}
        self._last_opt_1m_bar: Dict[str, Dict[str, float]] = {}
        self._prev_opt_5s_close: Dict[str, float] = {}

        # Open position tracking (for optional exits)
        self._open_contract: Optional[str] = None
        self._entry_delta_abs: Optional[float] = None
        self._entry_ts: Optional[datetime] = None

    # ---------- utilities ----------

    def _in_allowed_time(self, ts_utc: datetime) -> bool:
        if not self.allowed_sessions:
            return True
        lt = ts_utc.astimezone(self.tz).time()
        for a, b in self.allowed_sessions:
            if a <= lt <= b:
                return True
        return False

    def _round_atm(self, spot: float) -> int:
        step = max(int(self.strike_step), 1)
        return int(round(float(spot) / step) * step)

    def _candidate_strikes(self, atm: int) -> List[int]:
        step = max(int(self.strike_step), 1)
        w = max(int(self.atm_window_steps), 0)
        return [int(atm + k * step) for k in range(-w, w + 1)]

    def _ensure_tracker(self, symbol: str) -> _SeriesTracker:
        t = self._trackers.get(symbol)
        if t is None:
            t = _SeriesTracker(keep_minutes=max(30.0, self.delta_window_minutes * 6.0))
            self._trackers[symbol] = t
        return t

    def _update_ema_from_1h(self, bars: Any) -> None:
        fut_1h = None
        try:
            fut_1h = bars.get_futures("1h")
        except Exception:
            fut_1h = None
        if fut_1h is None or fut_1h.empty:
            return
        row = fut_1h.iloc[0]
        ts = row.get("timestamp")
        c = _safe_float(row.get("close"))
        if ts is None or c is None:
            return
        if self._last_1h_ts is not None and ts == self._last_1h_ts:
            return
        self._last_1h_ts = ts
        self._ema20_1h = _ema(self._ema20_1h, float(c), self.ema_fast_period)
        self._ema50_1h = _ema(self._ema50_1h, float(c), self.ema_slow_period)

    def _update_vwap_state(self, fut_close: float, fut_vol: float, base_seconds: float) -> None:
        # Store close*vol; if volume is missing/0, fall back to equal-weighted.
        vol = float(fut_vol) if np.isfinite(fut_vol) else 0.0
        self._fut_vwap_pxv.append(float(fut_close) * max(vol, 1.0))
        self._fut_vwap_v.append(max(vol, 1.0))

        max_len = int((self.vwap_window_minutes * 60.0) / max(base_seconds, 1.0)) + 2
        if len(self._fut_vwap_pxv) > max_len:
            self._fut_vwap_pxv = self._fut_vwap_pxv[-max_len:]
            self._fut_vwap_v = self._fut_vwap_v[-max_len:]

    def _vwap(self) -> Optional[float]:
        if not self._fut_vwap_pxv or not self._fut_vwap_v:
            return None
        v = float(np.sum(self._fut_vwap_v))
        if v <= 1e-9:
            return None
        return float(np.sum(self._fut_vwap_pxv) / v)

    def _direction_engine(self, snapshot: MarketSnapshot, bars: Any) -> Tuple[Optional[Direction], Dict[str, Any]]:
        if snapshot.futures_bar is None or snapshot.futures_bar.empty:
            return None, {"reason": "futures_missing"}

        fb = snapshot.futures_bar.iloc[0]
        fut_close = _safe_float(fb.get("close"))
        fut_vol = _safe_float(fb.get("volume")) or 0.0
        if fut_close is None:
            return None, {"reason": "futures_close_missing"}

        # update rolling vwap + EMA state
        base_seconds = float(getattr(bars, "base_bar_seconds", 5.0))
        self._update_vwap_state(float(fut_close), float(fut_vol), base_seconds=base_seconds)
        self._update_ema_from_1h(bars)

        vwap15 = self._vwap()
        if vwap15 is None or self._ema20_1h is None or self._ema50_1h is None:
            return None, {"reason": "direction_warmup", "vwap15": vwap15, "ema20": self._ema20_1h, "ema50": self._ema50_1h}

        bull = float(fut_close) > float(self._ema20_1h) > float(self._ema50_1h) and float(fut_close) > float(vwap15)
        bear = float(fut_close) < float(self._ema20_1h) < float(self._ema50_1h) and float(fut_close) < float(vwap15)

        meta = {"fut_close": float(fut_close), "vwap15": float(vwap15), "ema20_1h": float(self._ema20_1h), "ema50_1h": float(self._ema50_1h)}
        if bull:
            return "BULLISH", {**meta, "reason": "bullish_align"}
        if bear:
            return "BEARISH", {**meta, "reason": "bearish_align"}
        return None, {**meta, "reason": "no_trade_day"}

    def _nearest_expiry(self, chain: pd.DataFrame) -> Optional[Any]:
        if chain is None or chain.empty or "expiry" not in chain.columns:
            return None
        exps = [e for e in chain["expiry"].dropna().unique()]
        if not exps:
            return None
        return sorted(exps)[0]

    def _pick_window_rows(self, snapshot: MarketSnapshot, strikes: List[int], expiry: Optional[Any]) -> pd.DataFrame:
        if snapshot.options_chain is None or snapshot.options_chain.empty:
            return pd.DataFrame()
        ch = snapshot.options_chain
        out = ch.copy()
        out["strike"] = pd.to_numeric(out.get("strike"), errors="coerce")
        out = out.dropna(subset=["strike"])
        out = out[out["last"].fillna(0) > 0]
        out = out[out["strike"].astype(int).isin([int(x) for x in strikes])]
        if expiry is not None and "expiry" in out.columns and self.nearest_expiry_only:
            out = out[out["expiry"] == expiry]
        return out

    def _update_trackers_from_chain(self, now: datetime, rows: pd.DataFrame, bars: Any) -> None:
        if rows is None or rows.empty:
            return
        for _, r in rows.iterrows():
            sym = str(r.get("symbol", ""))
            if not sym:
                continue
            ltp = _safe_float(r.get("last"))
            oi = _safe_float(r.get("oi"))
            vol = _safe_float(r.get("volume"))

            # Prefer bar_cache option fields (may come from ticks); fallback to chain row.
            fields = None
            try:
                fields = bars.get_option_fields(sym)
            except Exception:
                fields = None
            delta = _safe_float((fields or {}).get("delta")) if fields else _safe_float(r.get("delta"))
            gamma = _safe_float((fields or {}).get("gamma")) if fields else _safe_float(r.get("gamma"))
            theta = _safe_float((fields or {}).get("theta")) if fields else _safe_float(r.get("theta"))
            iv = _safe_float((fields or {}).get("iv")) if fields else _safe_float(r.get("iv"))
            if ltp is None or ltp <= 0:
                continue

            self._ensure_tracker(sym).update(
                _Sample(
                    ts=now,
                    ltp=float(ltp),
                    oi=float(oi) if oi is not None else None,
                    volume=float(vol) if vol is not None else None,
                    delta=float(delta) if delta is not None else None,
                    gamma=float(gamma) if gamma is not None else None,
                    theta=float(theta) if theta is not None else None,
                    iv=float(iv) if iv is not None else None,
                )
            )

    def _deltas_5m(self, sym: str, now: datetime) -> Tuple[Optional[Dict[str, float]], Dict[str, Any]]:
        t = self._trackers.get(sym)
        if t is None:
            return None, {"reason": "no_tracker"}
        now_s = t.asof(now)
        if now_s is None:
            return None, {"reason": "no_now_sample"}
        past_ts = now - timedelta(minutes=float(self.delta_window_minutes))
        past_s = t.asof(past_ts)
        if past_s is None:
            return None, {"reason": "no_past_sample"}

        if now_s.ltp is None or past_s.ltp is None or past_s.ltp <= 0:
            return None, {"reason": "ltp_missing"}

        d_ltp = float(now_s.ltp - past_s.ltp)
        d_ltp_pct = (d_ltp / float(past_s.ltp)) * 100.0 if float(past_s.ltp) != 0 else 0.0

        d_oi = None
        d_oi_pct = None
        if now_s.oi is not None and past_s.oi is not None and past_s.oi > 0:
            d_oi = float(now_s.oi - past_s.oi)
            d_oi_pct = (d_oi / float(past_s.oi)) * 100.0

        d_vol = None
        d_vol_pct = None
        if now_s.volume is not None and past_s.volume is not None and past_s.volume > 0:
            d_vol = float(now_s.volume - past_s.volume)
            d_vol_pct = (d_vol / float(past_s.volume)) * 100.0

        out = {
            "d_ltp": d_ltp,
            "d_ltp_pct": d_ltp_pct,
        }
        if d_oi is not None and d_oi_pct is not None:
            out["d_oi"] = d_oi
            out["d_oi_pct"] = d_oi_pct
        if d_vol is not None and d_vol_pct is not None:
            out["d_vol"] = d_vol
            out["d_vol_pct"] = d_vol_pct
        meta = {"past_ts": past_s.ts, "now_ts": now_s.ts}
        return out, meta

    def _greeks_ok(self, sym: str, now: datetime) -> Tuple[bool, Dict[str, Any]]:
        t = self._trackers.get(sym)
        if t is None:
            return (not self.require_greeks), {"reason": "no_tracker"}
        s_now = t.asof(now)
        if s_now is None:
            return (not self.require_greeks), {"reason": "no_now_sample"}
        s_past = t.asof(now - timedelta(minutes=float(self.delta_window_minutes)))

        delta = s_now.delta
        gamma = s_now.gamma
        theta = s_now.theta
        iv = s_now.iv

        meta: Dict[str, Any] = {"delta": delta, "gamma": gamma, "theta": theta, "iv": iv}

        # Delta
        if delta is None or not np.isfinite(delta):
            return (not self.require_greeks), {**meta, "reason": "delta_missing"}
        if not (self.delta_min <= abs(float(delta)) <= self.delta_max):
            return False, {**meta, "reason": "delta_out_of_band"}

        # Gamma
        if gamma is None or not np.isfinite(gamma):
            return (not self.require_greeks), {**meta, "reason": "gamma_missing"}
        if float(gamma) < float(self.gamma_min):
            return False, {**meta, "reason": "gamma_too_low"}

        # Theta acceleration (optional but helpful)
        if theta is not None and s_past is not None and s_past.theta is not None and np.isfinite(theta) and np.isfinite(s_past.theta):
            accel = float(theta - s_past.theta)
            meta["theta_change_5m"] = accel
            if abs(accel) > float(self.theta_accel_max_abs) and accel < 0:
                return False, {**meta, "reason": "theta_accelerating"}

        # IV stability
        if iv is not None and s_past is not None and s_past.iv is not None and np.isfinite(iv) and np.isfinite(s_past.iv) and float(s_past.iv) > 0:
            iv_chg_pct = ((float(iv) - float(s_past.iv)) / float(s_past.iv)) * 100.0
            meta["iv_change_5m_pct"] = iv_chg_pct
            if abs(iv_chg_pct) > float(self.iv_change_max_pct):
                return False, {**meta, "reason": "iv_unstable"}

        return True, {**meta, "reason": "greeks_ok"}

    def _update_breakout_refs(self, snapshot: MarketSnapshot, bars: Any, opt_symbol: Optional[str]) -> None:
        # Futures 3m bar tracking (keep last two)
        try:
            fut3 = bars.get_futures("3min")
        except Exception:
            fut3 = None
        if fut3 is not None and not fut3.empty:
            r = fut3.iloc[0]
            ts = r.get("timestamp")
            if ts is not None and ts != self._last_3m_ts:
                self._prev_3m_bar = self._last_3m_bar
                self._last_3m_bar = {
                    "high": float(r.get("high", 0.0)),
                    "low": float(r.get("low", 0.0)),
                    "close": float(r.get("close", 0.0)),
                }
                self._last_3m_ts = ts

        # Option 1m bar tracking for candidate symbol (keep last two)
        if not opt_symbol:
            return
        try:
            opt1 = bars.get_option_premium(opt_symbol, "1min")
        except Exception:
            opt1 = None
        if opt1 is None or opt1.empty:
            return
        r = opt1.iloc[0]
        ts = r.get("timestamp")
        last_ts = self._last_opt_1m_ts.get(opt_symbol)
        if ts is not None and ts != last_ts:
            self._prev_opt_1m_bar[opt_symbol] = self._last_opt_1m_bar.get(opt_symbol, {})
            self._last_opt_1m_bar[opt_symbol] = {
                "high": float(r.get("high", 0.0)),
                "low": float(r.get("low", 0.0)),
                "close": float(r.get("close", 0.0)),
            }
            self._last_opt_1m_ts[opt_symbol] = ts

    # ---------- main loop ----------

    def generate_intents(
        self,
        snapshot: MarketSnapshot,
        context: Optional[Union[StrategyContext, Dict[str, Any]]] = None,
    ) -> List[Intent]:
        # Resolve context/bars
        bars = None
        ctx: Optional[StrategyContext] = None
        if isinstance(context, StrategyContext):
            ctx = context
            bars = context.bars
        elif isinstance(context, dict):
            bars = context.get("bars")

        if bars is None or not snapshot.has_spot():
            return []

        now = snapshot.timestamp

        # Time filter
        if not self._in_allowed_time(now):
            return []

        # Update direction engine state
        day_dir, dir_meta = self._direction_engine(snapshot, bars)
        if day_dir is None:
            # "NO TRADE DAY"
            return []

        # Optional exit management if position is open
        if self._open_contract is not None and ctx is not None and ctx.positions:
            # VWAP re-entry exit for bullish/bearish
            vwap15 = _safe_float(dir_meta.get("vwap15"))
            fut_close = _safe_float(dir_meta.get("fut_close"))
            if vwap15 is not None and fut_close is not None and vwap15 > 0:
                dist_pct = abs(float(fut_close) - float(vwap15)) / float(vwap15) * 100.0
                if dist_pct <= float(self.vwap_reentry_band_pct):
                    return [Intent(timestamp=now, direction="FLAT", action="EXIT", metadata={"reason": "vwap_reentry_exit", "dist_pct": dist_pct})]

            # Delta collapse exit (if we captured entry delta)
            if self._entry_delta_abs is not None:
                fields = None
                try:
                    fields = bars.get_option_fields(self._open_contract)
                except Exception:
                    fields = None
                cur_delta = _safe_float((fields or {}).get("delta")) if fields else None
                if cur_delta is not None:
                    cur_abs = abs(float(cur_delta))
                    collapse = (cur_abs / float(self._entry_delta_abs)) - 1.0 if float(self._entry_delta_abs) > 0 else 0.0
                    if collapse <= -0.15:
                        return [Intent(timestamp=now, direction="FLAT", action="EXIT", metadata={"reason": "delta_collapse_exit", "entry_abs": self._entry_delta_abs, "cur_abs": cur_abs})]

        # Emit pending entry on NEXT bar (avoid lookahead)
        if self._pending is not None:
            opt_type, meta = self._pending
            self._pending = None
            max_hold_bars = int(np.ceil((self.max_hold_minutes * 60.0) / max(float(getattr(bars, "base_bar_seconds", 5.0)), 1.0)))

            # risk-based targets
            sl = float(self.stop_loss_pct)
            tp1 = float(self.r_multiple_t1) * sl
            tp2 = float(self.r_multiple_t2) * sl

            meta_out = {
                **meta,
                "strategy": "institutional_flow_momentum",
                "direction": day_dir,
                "stop_loss_pct": sl,
                "take_profit_pct": tp2,  # compatibility: TP2
                "take_profit_pct_1": tp1,
                "take_profit_pct_2": tp2,
                "tp1_fraction": float(self.tp1_fraction),
                "max_hold_bars": int(max_hold_bars),
            }

            # Log once per trade signal (debug only, so tqdm is not broken)
            logger.debug(
                "InstitutionalFlowMomentum entry_signal=%s",
                {
                    "timestamp": now.isoformat(),
                    "direction": day_dir,
                    "instrument": opt_type,
                    "strike": meta_out.get("picked_strike"),
                    "entry": meta_out.get("entry_hint_ltp"),
                    "sl_pct": sl,
                    "t1_pct": tp1,
                    "t2_pct": tp2,
                    "oi_state": meta_out.get("oi_state"),
                    "delta": meta_out.get("delta"),
                },
            )

            return [
                Intent(
                    timestamp=now,
                    direction="LONG",
                    option_type=opt_type,  # type: ignore[arg-type]
                    size=int(max(1, self.contracts)),
                    metadata=meta_out,
                    reason_codes=["institutional_flow_breakout"],
                    tags=["institutional_flow_momentum", "breakout", day_dir.lower()],
                )
            ]

        # --- Build option universe window + update trackers (only small set) ---
        spot = snapshot.get_spot_price()
        if spot is None or not np.isfinite(spot):
            return []
        atm = self._round_atm(float(spot))
        strikes = self._candidate_strikes(atm)
        expiry = self._nearest_expiry(snapshot.options_chain) if snapshot.has_options() else None
        rows = self._pick_window_rows(snapshot, strikes=strikes, expiry=expiry)
        if rows.empty:
            return []

        self._update_trackers_from_chain(now, rows, bars)

        # Candidate selection aligned with selector(delta mode recommended in config)
        target_delta = float(self.target_delta)
        if day_dir == "BULLISH":
            cp = "C"
            opt_type: Literal["CALL", "PUT"] = "CALL"
            opp_cp = "P"
        else:
            cp = "P"
            opt_type = "PUT"
            opp_cp = "C"

        side_rows = rows[rows["cp"] == cp].copy()
        opp_rows = rows[rows["cp"] == opp_cp].copy()
        if side_rows.empty or opp_rows.empty:
            return []

        # choose best candidate in window by abs(delta-target), then oi/volume
        side_rows["abs_delta"] = (pd.to_numeric(side_rows.get("delta"), errors="coerce") - (target_delta if cp == "C" else -abs(target_delta))).abs()
        side_rows = side_rows.sort_values(by=["abs_delta", "oi", "volume", "symbol"], ascending=[True, False, False, True], kind="mergesort")
        cand = side_rows.iloc[0]
        sym = str(cand.get("symbol"))
        picked_strike = int(pd.to_numeric(cand.get("strike"), errors="coerce") or atm)

        # Update breakout reference bars for futures/option
        self._update_breakout_refs(snapshot, bars, opt_symbol=sym)

        # Institutional confirmation via 5m deltas
        deltas, deltas_meta = self._deltas_5m(sym, now)
        if deltas is None:
            return []

        d_ltp_pct = float(deltas.get("d_ltp_pct", 0.0))
        d_oi_pct = _safe_float(deltas.get("d_oi_pct"))
        d_vol_pct = _safe_float(deltas.get("d_vol_pct"))

        if self.require_oi_confirmation and (d_oi_pct is None or not np.isfinite(d_oi_pct)):
            return []
        if self.require_volume_confirmation and (d_vol_pct is None or not np.isfinite(d_vol_pct)):
            return []

        if d_ltp_pct < float(self.min_premium_change_pct):
            return []
        if self.require_oi_confirmation and d_oi_pct is not None and float(d_oi_pct) < float(self.min_oi_change_pct):
            return []
        if self.require_volume_confirmation and d_vol_pct is not None and float(d_vol_pct) < float(self.min_volume_change_pct):
            return []

        # Opposite weakness check (any in window)
        opp_ok = False
        for _, r in opp_rows.iterrows():
            osym = str(r.get("symbol", ""))
            if not osym:
                continue
            od, _ = self._deltas_5m(osym, now)
            if od is None:
                continue
            od_ltp_pct = float(od.get("d_ltp_pct", 0.0))
            od_oi_pct = _safe_float(od.get("d_oi_pct"))
            if (od_oi_pct is not None and od_oi_pct <= -abs(float(self.opposite_weak_threshold_pct))) or (od_ltp_pct <= -abs(float(self.opposite_weak_threshold_pct))):
                opp_ok = True
                break
        if not opp_ok:
            return []

        # Greeks stability (candidate symbol)
        g_ok, g_meta = self._greeks_ok(sym, now)
        if not g_ok:
            return []

        # Breakout triggers (use PREVIOUS closed bars, tracked in state)
        if self._prev_3m_bar is None:
            return []
        prev3_hi = float(self._prev_3m_bar.get("high", 0.0))
        prev3_lo = float(self._prev_3m_bar.get("low", 0.0))
        if prev3_hi <= 0 and prev3_lo <= 0:
            return []

        fut5 = bars.get_futures(str(getattr(bars, "base_bar_size", "5s")))
        if fut5 is None or fut5.empty:
            return []
        fut5c = _safe_float(fut5.iloc[0].get("close"))
        if fut5c is None:
            return []

        opt5 = bars.get_option_premium(sym, str(getattr(bars, "base_bar_size", "5s")))
        if opt5 is None or opt5.empty:
            return []
        opt5c = _safe_float(opt5.iloc[0].get("close"))
        if opt5c is None or opt5c <= 0:
            return []

        # Capture previous values for impulse check, then commit latest before any return below.
        prev_opt_5s_close = self._prev_opt_5s_close.get(sym)
        prev_fut_5s_close = self._fut_prev_close

        def _commit_prev() -> None:
            self._prev_opt_5s_close[sym] = float(opt5c)
            self._fut_prev_close = float(fut5c)

        prev1 = self._prev_opt_1m_bar.get(sym)
        if not prev1 or "high" not in prev1 or "low" not in prev1:
            _commit_prev()
            return []
        prev1_hi = float(prev1.get("high", 0.0))
        prev1_lo = float(prev1.get("low", 0.0))
        if prev1_hi <= 0 or prev1_lo <= 0:
            _commit_prev()
            return []

        if day_dir == "BULLISH":
            fut_break = float(fut5c) > prev3_hi
            opt_break = float(opt5c) > prev1_hi
        else:
            fut_break = float(fut5c) < prev3_lo
            opt_break = float(opt5c) < prev1_lo
        if not (fut_break and opt_break):
            _commit_prev()
            return []

        # Aggressive buyers proxy (LTP impulse vs prev 5s close)
        if self.use_aggressive_proxy:
            if prev_opt_5s_close is not None and prev_opt_5s_close > 0:
                imp = (float(opt5c) / float(prev_opt_5s_close) - 1.0) * 100.0
                if abs(imp) < float(self.min_opt_impulse_pct_5s):
                    _commit_prev()
                    return []
            if prev_fut_5s_close is not None and prev_fut_5s_close > 0:
                impf = (float(fut5c) / float(prev_fut_5s_close) - 1.0) * 100.0
                if abs(impf) < float(self.min_fut_impulse_pct_5s):
                    _commit_prev()
                    return []

        # update prev closes
        _commit_prev()

        # schedule entry next bar
        oi_state = "LONG_BUILDUP"
        meta = {
            "signal_ts": now,
            "direction_engine": dir_meta,
            "picked_symbol": sym,
            "picked_strike": int(picked_strike),
            "picked_expiry": str(cand.get("expiry")),
            "atm": int(atm),
            "strikes_window": [int(x) for x in strikes],
            "oi_state": oi_state,
            "deltas_5m": deltas,
            "deltas_5m_meta": deltas_meta,
            "greeks": g_meta,
            "delta": g_meta.get("delta"),
            "entry_hint_ltp": float(opt5c),
            "breakout": {
                "prev_3m_hi": prev3_hi,
                "prev_3m_lo": prev3_lo,
                "prev_1m_hi": prev1_hi,
                "prev_1m_lo": prev1_lo,
                "fut_5s_close": float(fut5c),
                "opt_5s_close": float(opt5c),
            },
        }
        self._pending = (opt_type, meta)
        return []

    def on_trade_filled(self, intent: Intent, contract: str, fill_price: float, quantity: int):
        # Capture entry greeks for optional delta-collapse exit logic.
        self._open_contract = str(contract)
        self._entry_ts = intent.timestamp
        self._entry_delta_abs = None
        try:
            delta = _safe_float((intent.metadata or {}).get("delta"))
            if delta is not None:
                self._entry_delta_abs = abs(float(delta))
        except Exception:
            self._entry_delta_abs = None

    def on_trade_closed(self, intent: Intent, contract: str, pnl: float):
        self._open_contract = None
        self._entry_delta_abs = None
        self._entry_ts = None


