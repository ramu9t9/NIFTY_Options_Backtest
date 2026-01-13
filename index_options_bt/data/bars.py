"""
Bar building utilities for converting tick data to OHLCV bars and option chain snapshots.

Supports:
- OHLCV bar building for index/futures from ltp ticks
- As-of option chain snapshots (carry forward last-known values)
"""

from datetime import datetime
from typing import Optional, Literal
import pandas as pd
import numpy as np

# Default aggregation methods
DEFAULT_VOLUME_AGG = "max"  # Safer for snapshot semantics
DEFAULT_OI_AGG = "last"


def _bar_volume(
    df: pd.DataFrame,
    resampled: pd.core.resample.Resampler,
    mode: Literal["incremental", "cumulative"],
    target_index: pd.Index,
) -> pd.Series:
    """Compute bar volume from tick volume series with well-defined semantics."""
    if "volume" not in df.columns:
        return pd.Series(0.0, index=target_index)
    if mode == "cumulative":
        v_last = resampled["volume"].last()
        v_first = resampled["volume"].first()
        out = (pd.to_numeric(v_last, errors="coerce").fillna(0.0) - pd.to_numeric(v_first, errors="coerce").fillna(0.0)).clip(lower=0.0)
        return out.reindex(target_index).fillna(0.0)
    # incremental
    out = pd.to_numeric(resampled["volume"].sum(), errors="coerce").fillna(0.0)
    return out.reindex(target_index).fillna(0.0)


def build_ohlcv(
    ticks_df: pd.DataFrame,
    bar_size: str,
    price_col: str = "ltp",
    volume_agg: Literal["max", "sum", "last", "first"] = DEFAULT_VOLUME_AGG,
    oi_agg: Literal["max", "sum", "last", "first"] = DEFAULT_OI_AGG,
    volume_mode: Literal["incremental", "cumulative"] = "incremental",
) -> pd.DataFrame:
    """
    Build OHLCV bars from tick data.
    
    Args:
        ticks_df: DataFrame with columns: ts, ltp, volume, oi (and optionally bid, ask)
        bar_size: Pandas frequency string (e.g., "5s", "15s", "30s")
        price_col: Column name for price data (default "ltp")
        volume_agg: Aggregation method for volume (default "max")
        oi_agg: Aggregation method for OI (default "last")
        
    Returns:
        DataFrame indexed by bar_end_ts_utc with columns: open, high, low, close, volume, oi
        
    Example:
        >>> ticks = pd.DataFrame({
        ...     "ts": pd.date_range("2025-10-01 09:15:00", periods=100, freq="1s", tz="UTC"),
        ...     "ltp": np.random.uniform(19000, 20000, 100),
        ...     "volume": [0] * 100,  # Index has volume=0
        ...     "oi": [None] * 100,
        ... })
        >>> bars = build_ohlcv(ticks, "15s", price_col="ltp")
        >>> bars.index.name
        'bar_end_ts_utc'
    """
    if ticks_df.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "oi"],
            index=pd.DatetimeIndex([], name="bar_end_ts_utc", tz="UTC"),
        )
    
    # Ensure ts is datetime and set as index temporarily
    df = ticks_df.copy()
    if "ts" not in df.columns:
        raise ValueError("ticks_df must have 'ts' column")
    
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
    df = df.set_index("ts").sort_index()
    
    # Resample to bar_size frequency
    # Use right label to get bar end timestamps
    resampled = df.resample(bar_size, label="right", closed="right")
    
    # Build OHLC from price_col (pandas-compatible)
    ohlc = resampled[price_col].ohlc()
    
    # Aggregate volume and OI
    # NOTE: volume_agg is legacy; volume_mode defines semantics and is what the engine uses.
    volume_data = _bar_volume(df, resampled, mode=volume_mode, target_index=ohlc.index)
    
    oi_data = None
    if "oi" in df.columns:
        if oi_agg == "max":
            oi_data = resampled["oi"].max()
        elif oi_agg == "sum":
            oi_data = resampled["oi"].sum()
        elif oi_agg == "last":
            oi_data = resampled["oi"].last()
        elif oi_agg == "first":
            oi_data = resampled["oi"].first()
        else:
            oi_data = resampled["oi"].last()
    else:
        oi_data = pd.Series(None, index=ohlc.index, dtype=float)
    
    # Combine
    bars_df = pd.DataFrame({
        "open": ohlc["open"],
        "high": ohlc["high"],
        "low": ohlc["low"],
        "close": ohlc["close"],
        "volume": volume_data,
        "oi": oi_data,
    })
    
    bars_df.index.name = "bar_end_ts_utc"
    return bars_df


def build_option_asof_snapshot(
    ticks_df: pd.DataFrame,
    bar_ts_index: pd.DatetimeIndex,
    bar_size: str,
    volume_mode: Literal["incremental", "cumulative"] = "incremental",
) -> pd.DataFrame:
    """
    Build option chain snapshot at each bar timestamp using as-of semantics.
    
    For each option symbol, carry forward the last-known values (bid, ask, ltp, greeks, etc.)
    to each bar timestamp. This simulates "as-of" data availability.
    
    Args:
        ticks_df: DataFrame with columns: ts, symbol, bid, ask, ltp, iv, delta, gamma, theta, vega, volume, oi
        bar_ts_index: DatetimeIndex of bar end timestamps (UTC)
        
    Returns:
        DataFrame with columns:
        timestamp, symbol, expiry, strike, cp, bid, ask, last, iv, delta, gamma, theta, vega, volume, oi
        Indexed by timestamp (bar end) with multi-index for symbol
        
    Example:
        >>> ticks = pd.DataFrame({
        ...     "ts": [...],
        ...     "symbol": ["NIFTY25NOV2526000CE"] * 100,
        ...     "ltp": [...],
        ...     "bid": [...],
        ...     "ask": [...],
        ... })
        >>> bar_ts = pd.date_range("2025-10-01 09:15:15", periods=10, freq="15s", tz="UTC")
        >>> chain = build_option_asof_snapshot(ticks, bar_ts, bar_size="15s")
    """
    if ticks_df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp", "symbol", "expiry", "strike", "cp",
                "bid", "ask", "last", "iv", "delta", "gamma", "theta", "vega",
                "volume", "oi",
            ],
        )
    
    from .symbols import parse_symbol
    
    df = ticks_df.copy()
    
    # Ensure ts is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    
    df = df.sort_values(["symbol", "ts"])
    
    # Parse symbols to extract expiry, strike, cp
    parsed_symbols = []
    for symbol in df["symbol"].unique():
        try:
            parsed = parse_symbol(symbol)
            parsed_symbols.append({
                "symbol": symbol,
                "expiry": parsed.expiry,
                "strike": parsed.strike,
                "cp": parsed.cp,
            })
        except ValueError:
            # Skip invalid symbols
            continue
    
    if not parsed_symbols:
        return pd.DataFrame(
            columns=[
                "timestamp", "symbol", "expiry", "strike", "cp",
                "bid", "ask", "last", "iv", "delta", "gamma", "theta", "vega",
                "volume", "oi",
            ],
        )
    
    symbol_map = pd.DataFrame(parsed_symbols).set_index("symbol")
    
    # Efficient as-of join per symbol using merge_asof
    chain_parts = []
    bars_df = pd.DataFrame({"timestamp": bar_ts_index})
    bars_df = bars_df.sort_values("timestamp")

    keep_cols = ["ts", "symbol", "bid", "ask", "ltp", "iv", "delta", "gamma", "theta", "vega", "volume", "oi"]
    for symbol in symbol_map.index:
        symbol_ticks = df[df["symbol"] == symbol][keep_cols].copy()
        if symbol_ticks.empty:
            continue
        symbol_ticks = symbol_ticks.sort_values("ts")
        merged = pd.merge_asof(
            bars_df,
            symbol_ticks,
            left_on="timestamp",
            right_on="ts",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=["ts"])
        # drop bars before first tick for symbol
        merged = merged[merged["symbol"].notna()]
        if merged.empty:
            continue

        # Replace carried-forward volume with per-bar volume.
        # This is critical for liquidity filters and strategy features.
        if "volume" in symbol_ticks.columns:
            vdf = symbol_ticks[["ts", "volume"]].copy()
            vdf["ts"] = pd.to_datetime(vdf["ts"], utc=True)
            vdf = vdf.set_index("ts").sort_index()
            r = vdf.resample(bar_size, label="right", closed="right")
            if volume_mode == "cumulative":
                vbar = (r["volume"].last().fillna(0.0) - r["volume"].first().fillna(0.0)).clip(lower=0.0)
            else:
                vbar = r["volume"].sum().fillna(0.0)
            vbar = vbar.rename("volume_bar").reset_index().rename(columns={"ts": "timestamp"})
            merged = merged.merge(vbar, on="timestamp", how="left")
            if "volume_bar" in merged.columns:
                merged["volume"] = merged["volume_bar"].fillna(0.0)
                merged = merged.drop(columns=["volume_bar"])

        parsed = symbol_map.loc[symbol]
        merged["symbol"] = symbol
        merged["expiry"] = parsed["expiry"]
        merged["strike"] = parsed["strike"]
        merged["cp"] = parsed["cp"]
        merged = merged.rename(columns={"ltp": "last"})
        chain_parts.append(merged)

    if not chain_parts:
        return pd.DataFrame(
            columns=[
                "timestamp", "symbol", "expiry", "strike", "cp",
                "bid", "ask", "last", "iv", "delta", "gamma", "theta", "vega",
                "volume", "oi",
            ],
        )

    chain_df = pd.concat(chain_parts, ignore_index=True)
    
    # Ensure timestamp is UTC
    if not pd.api.types.is_datetime64_any_dtype(chain_df["timestamp"]):
        chain_df["timestamp"] = pd.to_datetime(chain_df["timestamp"], utc=True)
    
    return chain_df

