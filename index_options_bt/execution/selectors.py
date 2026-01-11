"""
Contract selectors: ATM, DTE, Delta with liquidity/expiry/strike window filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config.schemas import SelectorConfig
from ..data.models import MarketSnapshot
from ..strategy.base import Intent
from .chain_cache import ChainCache
from .contracts import ContractSelector, OptionContract, SelectedLeg


def _is_weekly_expiry(expiry: date) -> bool:
    # NIFTY weekly: Thu. Monthly: last Thu (we approximate monthly as last Thu of month).
    return expiry.weekday() == 3  # Thursday


def _is_monthly_expiry(expiry: date) -> bool:
    if expiry.weekday() != 3:
        return False
    # last Thu of month: add 7 days and month changes
    return (expiry + pd.Timedelta(days=7)).month != expiry.month


def _expiry_ok(expiry: date, pref: str) -> bool:
    if pref == "nearest":
        return True
    if pref == "weekly":
        return _is_weekly_expiry(expiry)
    if pref == "monthly":
        return _is_monthly_expiry(expiry)
    return True


def _apply_strike_window(df: pd.DataFrame, spot: float, cfg: SelectorConfig) -> pd.DataFrame:
    if df.empty:
        return df
    if cfg.strike_window is None:
        return df
    sw = cfg.strike_window
    if sw.kind == "pct":
        lo = spot * (1.0 - sw.value)
        hi = spot * (1.0 + sw.value)
        return df[(df["strike"].astype(float) >= lo) & (df["strike"].astype(float) <= hi)]
    # strikes count: approximate by abs(strike-spot) <= N*step; step inferred from chain strikes
    strikes = pd.to_numeric(df["strike"], errors="coerce").dropna().astype(int).unique()
    if len(strikes) >= 2:
        step = int(np.median(np.diff(np.sort(strikes))))
        step = max(step, 1)
    else:
        step = 50
    lo = spot - sw.value * step
    hi = spot + sw.value * step
    return df[(df["strike"].astype(float) >= lo) & (df["strike"].astype(float) <= hi)]


def _apply_liquidity(df: pd.DataFrame, cfg: SelectorConfig) -> pd.DataFrame:
    """
    Apply liquidity filters. 
    
    IMPORTANT: Like the old backtest engine, we primarily use LTP (last) for pricing.
    If bid/ask are 0/NaN but LTP exists, we allow the contract (execution will use LTP).
    """
    if df.empty:
        return df
    liq = cfg.liquidity
    out = df.copy()
    
    # Price filter: require either bid >= min_bid OR last (LTP) > 0
    # This matches old engine behavior which uses LTP directly
    if float(liq.min_bid) > 0:
        has_price = (out["bid"].fillna(0) >= float(liq.min_bid)) | (out["last"].fillna(0) > 0)
        out = out[has_price]
    else:
        # Even if min_bid=0, still require at least LTP to exist
        out = out[out["last"].fillna(0) > 0]
    
    # Spread filter: if spread is NaN (bid/ask both 0), allow if LTP exists
    # This is realistic - many options have 0 bid/ask but trade on LTP
    if float(liq.max_spread_pct) < np.inf:
        spread_ok = (out["spread_pct"].fillna(np.inf) <= float(liq.max_spread_pct)) | (
            out["spread_pct"].isna() & (out["last"].fillna(0) > 0)
        )
        out = out[spread_ok]
    
    # OI and volume filters (usually 0, so no filtering)
    out = out[out["oi"].fillna(0) >= float(liq.min_oi)]
    out = out[out["volume"].fillna(0) >= float(liq.min_volume)]
    return out


def _direction_to_cp(intent: Intent) -> Optional[str]:
    if intent.direction == "FLAT":
        return None
    if intent.option_type == "CALL":
        return "C"
    if intent.option_type == "PUT":
        return "P"
    return None


def _deterministic_sort(
    df: pd.DataFrame,
    spot: float,
    target_delta: Optional[float] = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    # tie-break order:
    # (expiry asc, spread_pct asc, abs(strike-spot) asc, abs(delta-target) asc, oi desc, volume desc, symbol asc)
    out = df.copy()
    out["abs_strike_spot"] = (out["strike"].astype(float) - float(spot)).abs()
    if target_delta is None or not np.isfinite(target_delta):
        out["abs_delta_target"] = np.nan
    else:
        out["abs_delta_target"] = (pd.to_numeric(out["delta"], errors="coerce") - float(target_delta)).abs()

    # expiry sort: None goes last
    out["expiry_sort"] = out["expiry"].apply(lambda d: d.toordinal() if isinstance(d, date) else 10**12)

    out = out.sort_values(
        by=[
            "expiry_sort",
            "spread_pct",
            "abs_strike_spot",
            "abs_delta_target",
            "oi",
            "volume",
            "symbol",
        ],
        ascending=[True, True, True, True, False, False, True],
        kind="mergesort",
    )
    return out.drop(columns=["abs_strike_spot", "abs_delta_target", "expiry_sort"], errors="ignore")


@dataclass
class _BaseSelector(ContractSelector):
    chain_cache: ChainCache
    contract_multiplier: int = 1

    def _prep(self, snapshot: MarketSnapshot, cfg: SelectorConfig, intent: Intent) -> Tuple[pd.DataFrame, float, str]:
        spot = snapshot.get_spot_price()
        if spot is None:
            return pd.DataFrame(), np.nan, ""
        cp = _direction_to_cp(intent)
        if cp is None:
            return pd.DataFrame(), float(spot), ""

        chain = self.chain_cache.get_or_build(snapshot.options_chain, snapshot.timestamp)
        df = chain.copy()

        # expiry preference filter
        df = df[df["expiry"].apply(lambda d: _expiry_ok(d, cfg.expiry_preference) if isinstance(d, date) else False)]
        # cp filter
        df = df[df["cp"] == cp]
        # strike window
        df = _apply_strike_window(df, float(spot), cfg)
        # liquidity
        df = _apply_liquidity(df, cfg)

        return df, float(spot), cp

    def _make_leg(self, row: pd.Series, intent: Intent) -> SelectedLeg:
        expiry = row["expiry"]
        strike = int(row["strike"])
        cp = str(row["cp"])
        contract = OptionContract(
            symbol=str(row["symbol"]),
            expiry=expiry,
            strike=strike,
            cp=cp,  # type: ignore[arg-type]
            multiplier=int(self.contract_multiplier),
        )
        side = "BUY" if intent.direction == "LONG" else "SELL"
        return SelectedLeg(contract=contract, side=side, qty=int(intent.size), tag=None)


@dataclass
class ATMSelector(_BaseSelector):
    """Pick ATM strike for nearest expiry (subject to filters)."""

    def select(self, snapshot: MarketSnapshot, intent: Intent, selector_cfg: SelectorConfig) -> Sequence[SelectedLeg]:
        df, spot, _ = self._prep(snapshot, selector_cfg, intent)
        if df.empty or not np.isfinite(spot):
            return []
        df = _deterministic_sort(df, spot)
        row = df.iloc[0]
        return [self._make_leg(row, intent)]


@dataclass
class DTESelector(_BaseSelector):
    """Pick expiry closest to target_dte then best strike within expiry."""

    def select(self, snapshot: MarketSnapshot, intent: Intent, selector_cfg: SelectorConfig) -> Sequence[SelectedLeg]:
        if selector_cfg.target_dte is None:
            return []
        df, spot, _ = self._prep(snapshot, selector_cfg, intent)
        if df.empty or not np.isfinite(spot):
            return []
        target = float(selector_cfg.target_dte)
        df = df.copy()
        df["abs_dte"] = (pd.to_numeric(df["dte"], errors="coerce") - target).abs()
        # First choose min abs_dte expiry group (deterministic via expiry asc)
        df["expiry_sort"] = df["expiry"].apply(lambda d: d.toordinal() if isinstance(d, date) else 10**12)
        df = df.sort_values(by=["abs_dte", "expiry_sort"], ascending=[True, True], kind="mergesort")
        best_expiry = df.iloc[0]["expiry"]
        df = df[df["expiry"] == best_expiry]
        df = _deterministic_sort(df, spot)
        row = df.iloc[0]
        return [self._make_leg(row, intent)]


@dataclass
class DeltaSelector(_BaseSelector):
    """Pick contract with delta closest to target_delta (subject to filters)."""

    def select(self, snapshot: MarketSnapshot, intent: Intent, selector_cfg: SelectorConfig) -> Sequence[SelectedLeg]:
        if selector_cfg.target_delta is None:
            return []
        df, spot, _ = self._prep(snapshot, selector_cfg, intent)
        if df.empty or not np.isfinite(spot):
            return []
        # For puts, deltas are typically negative. User configs commonly specify target_delta as a
        # positive magnitude (e.g., 0.25). Convert to the appropriate sign.
        target_in = float(selector_cfg.target_delta)
        cp = _direction_to_cp(intent)
        if cp == "P":
            target = -abs(target_in)
        else:
            target = abs(target_in)
        df = df.copy()
        df = _deterministic_sort(df, spot, target_delta=target)
        row = df.iloc[0]
        return [self._make_leg(row, intent)]


def build_selector(cfg: SelectorConfig, chain_cache: ChainCache, contract_multiplier: int = 1) -> ContractSelector:
    if cfg.mode == "atm":
        return ATMSelector(chain_cache=chain_cache, contract_multiplier=contract_multiplier)
    if cfg.mode == "dte":
        return DTESelector(chain_cache=chain_cache, contract_multiplier=contract_multiplier)
    if cfg.mode == "delta":
        return DeltaSelector(chain_cache=chain_cache, contract_multiplier=contract_multiplier)
    return ATMSelector(chain_cache=chain_cache, contract_multiplier=contract_multiplier)


