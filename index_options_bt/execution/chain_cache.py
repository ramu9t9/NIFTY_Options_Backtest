"""
Option chain normalization + derived fields + lightweight caching.

Input chain (from SQLite provider snapshot) is expected to include:
timestamp, symbol, expiry, strike, cp, bid, ask, last, iv, delta, gamma, theta, vega, volume, oi
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _to_date(x) -> Optional[date]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    # pandas Timestamp
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


def normalize_chain(chain: pd.DataFrame, ts_utc: datetime) -> pd.DataFrame:
    """Normalize chain to stable schema and add derived fields."""
    if chain is None or chain.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "expiry",
                "strike",
                "cp",
                "bid",
                "ask",
                "last",
                "iv",
                "delta",
                "gamma",
                "theta",
                "vega",
                "volume",
                "oi",
                "mid",
                "spread_pct",
                "dte",
            ]
        )

    df = chain.copy()

    # Ensure required columns exist
    for col in [
        "timestamp",
        "symbol",
        "expiry",
        "strike",
        "cp",
        "bid",
        "ask",
        "last",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "volume",
        "oi",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    # expiry can be datetime/date; normalize to date
    df["expiry"] = df["expiry"].apply(_to_date)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype("Int64")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["last"] = pd.to_numeric(df["last"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")

    # mid + spread%
    df["mid"] = np.where(
        np.isfinite(df["bid"]) & np.isfinite(df["ask"]),
        (df["bid"] + df["ask"]) / 2.0,
        df["last"],
    )
    spread = df["ask"] - df["bid"]
    denom = df["mid"].replace(0, np.nan)
    df["spread_pct"] = (spread / denom).replace([np.inf, -np.inf], np.nan)

    # dte: days to expiry (calendar days)
    ts_date = ts_utc.date()
    df["dte"] = df["expiry"].apply(lambda d: (d - ts_date).days if isinstance(d, date) else np.nan)

    # keep only rows for ts_utc (snapshot should already be filtered but be safe)
    ts_key = pd.Timestamp(ts_utc)
    if ts_key.tzinfo is None:
        ts_key = ts_key.tz_localize("UTC")
    else:
        ts_key = ts_key.tz_convert("UTC")
    df = df[df["timestamp"] == ts_key]

    return df.reset_index(drop=True)


@dataclass
class ChainCache:
    """
    Cache normalized chains by timestamp.

    Key is timestamp UTC; value is normalized DataFrame for that timestamp.
    """

    _cache: Dict[pd.Timestamp, pd.DataFrame]

    def __init__(self) -> None:
        self._cache = {}

    def get(self, ts_utc: datetime) -> Optional[pd.DataFrame]:
        key = pd.Timestamp(ts_utc)
        if key.tzinfo is None:
            key = key.tz_localize("UTC")
        else:
            key = key.tz_convert("UTC")
        return self._cache.get(key)

    def put(self, ts_utc: datetime, df: pd.DataFrame) -> None:
        key = pd.Timestamp(ts_utc)
        if key.tzinfo is None:
            key = key.tz_localize("UTC")
        else:
            key = key.tz_convert("UTC")
        self._cache[key] = df

    def get_or_build(self, chain: pd.DataFrame, ts_utc: datetime) -> pd.DataFrame:
        cached = self.get(ts_utc)
        if cached is not None:
            return cached
        df = normalize_chain(chain, ts_utc)
        self.put(ts_utc, df)
        return df


