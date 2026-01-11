"""
SQLite data provider for ltp_ticks table.

Loads data from SQLite database with session chunking and caching for performance.
"""

import sqlite3
from datetime import datetime, time, timezone, timedelta
from typing import Optional, Dict, Tuple, Iterator
import pandas as pd
import numpy as np

from .models import DataProvider, MarketSnapshot
from .bars import build_ohlcv, build_option_asof_snapshot
from .calendars import generate_session_windows, generate_bar_timestamps, IST, UTC, DEFAULT_SESSION_START_IST, DEFAULT_SESSION_END_IST
from .symbols import is_index_symbol, is_future_symbol, is_option_symbol

# Cache key: (session_date_ist_str, bar_size)
_SessionCacheKey = Tuple[str, str]


class SQLiteDataProvider:
    """
    SQLite data provider with session-based chunking and caching.
    
    Loads data from ltp_ticks table, builds OHLCV bars and option chain snapshots
    at bar_size frequency. Uses session chunking to manage memory efficiently.
    """
    
    def __init__(
        self,
        sqlite_path: str,
        table: str = "ltp_ticks",
        tz_display: timezone | str = IST,
        bar_size: str = "15s",
        symbol_index: str = "NIFTY 50",
        chunking: str = "session",
        session_start_ist: time | str = DEFAULT_SESSION_START_IST,
        session_end_ist: time | str = DEFAULT_SESSION_END_IST,
    ):
        """
        Initialize SQLite data provider.
        
        Args:
            sqlite_path: Path to SQLite database file
            table: Table name (default "ltp_ticks")
            tz_display: Display timezone (default IST) for session date extraction
            bar_size: Bar size frequency (e.g., "5s", "15s", "30s")
            symbol_index: Index symbol (default "NIFTY 50")
            chunking: Chunking strategy (default "session")
            session_start_ist: Market session start time in IST (default 09:15)
            session_end_ist: Market session end time in IST (default 15:30)
        """
        self.sqlite_path = sqlite_path
        self.table = table
        # Normalize timezone display
        if isinstance(tz_display, str):
            # Only IST supported for now; keep fixed-offset IST (UTC+5:30)
            self.tz_display = IST if tz_display.lower() in ("asia/kolkata", "ist") else IST
        else:
            self.tz_display = tz_display
        self.bar_size = bar_size
        self.symbol_index = symbol_index
        self.chunking = chunking
        self.session_start_ist = session_start_ist
        self.session_end_ist = session_end_ist
        
        # Session cache: keyed by (session_date_ist_str, bar_size)
        self._cache: Dict[_SessionCacheKey, Dict[str, pd.DataFrame]] = {}
        
        # Connection (keep alive for performance)
        self._conn: Optional[sqlite3.Connection] = None
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            self._conn.execute("PRAGMA busy_timeout=5000;")
        return self._conn
    
    def close(self):
        """Close database connection"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _get_session_date_ist(self, ts_utc: datetime) -> str:
        """Extract session date (IST) as string for cache key"""
        ts_ist = ts_utc.astimezone(self.tz_display)
        return ts_ist.date().isoformat()
    
    def _load_session_data(
        self,
        session_start_utc: datetime,
        session_end_utc: datetime,
    ) -> pd.DataFrame:
        """
        Load all tick data for a trading session.
        
        Args:
            session_start_utc: Session start in UTC
            session_end_utc: Session end in UTC
            
        Returns:
            DataFrame with columns: id, symbol, token, ts, ltp, bid, ask, volume, oi, delta, gamma, theta, vega, iv, source
        """
        conn = self._get_connection()
        
        # Query ticks for this session
        # Use ORDER BY ts ASC, id ASC for deterministic ordering
        query = f"""
            SELECT id, symbol, token, ts, ltp, bid, ask, volume, oi,
                   delta, gamma, theta, vega, iv, source
            FROM {self.table}
            WHERE ts >= ? AND ts < ?
            ORDER BY ts ASC, id ASC
        """
        
        start_iso = session_start_utc.isoformat()
        end_iso = session_end_utc.isoformat()
        
        df = pd.read_sql_query(query, conn, params=(start_iso, end_iso))
        
        if df.empty:
            return df
        
        # Parse ts column to datetime
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        
        return df
    
    def _build_session_bars(
        self,
        ticks_df: pd.DataFrame,
        session_start_utc: datetime,
        session_end_utc: datetime,
    ) -> Dict[str, pd.DataFrame]:
        """
        Build bars and option chain snapshots for a session.
        
        Returns:
            Dict with keys: "spot_bars", "futures_bars", "bar_timestamps", "option_chain"
        """
        # Generate bar timestamps for this session
        bar_timestamps = generate_bar_timestamps(session_start_utc, session_end_utc, self.bar_size)
        
        if bar_timestamps.empty:
            return {
                "spot_bars": pd.DataFrame(),
                "futures_bars": pd.DataFrame(),
                "bar_timestamps": bar_timestamps,
                "option_chain": pd.DataFrame(),
            }
        
        # Split ticks by symbol type
        index_ticks = ticks_df[ticks_df["symbol"] == self.symbol_index].copy()
        fut_ticks = ticks_df[ticks_df["symbol"].apply(is_future_symbol)].copy()
        opt_ticks = ticks_df[ticks_df["symbol"].apply(is_option_symbol)].copy()
        
        # Build spot/index bars
        spot_bars = pd.DataFrame()
        if not index_ticks.empty:
            spot_bars = build_ohlcv(index_ticks, self.bar_size, price_col="ltp")
        
        # Build futures bars (take first FUT symbol if present)
        futures_bars = pd.DataFrame()
        if not fut_ticks.empty:
            # Use the most liquid FUT (highest volume) or first available
            fut_symbols = fut_ticks["symbol"].unique()
            if len(fut_symbols) > 0:
                primary_fut = fut_symbols[0]  # Could enhance to pick most liquid
                primary_fut_ticks = fut_ticks[fut_ticks["symbol"] == primary_fut]
                if not primary_fut_ticks.empty:
                    futures_bars = build_ohlcv(primary_fut_ticks, self.bar_size, price_col="ltp")
        
        # Build option chain snapshots (as-of)
        option_chain = pd.DataFrame()
        if not opt_ticks.empty:
            option_chain = build_option_asof_snapshot(opt_ticks, bar_timestamps)
        
        return {
            "spot_bars": spot_bars,
            "futures_bars": futures_bars,
            "bar_timestamps": bar_timestamps,
            "option_chain": option_chain,
        }
    
    def _get_session_cache_key(self, ts_utc: datetime) -> _SessionCacheKey:
        """Get cache key for a timestamp's session"""
        session_date = self._get_session_date_ist(ts_utc)
        return (session_date, self.bar_size)
    
    def _ensure_session_cached(self, ts_utc: datetime):
        """
        Ensure the session containing ts_utc is loaded and cached.
        
        Loads session data, builds bars, and caches the results.
        """
        # Determine which session this timestamp belongs to
        session_windows = generate_session_windows(
            start=ts_utc - timedelta(days=1),  # Look back a bit to find session
            end=ts_utc + timedelta(days=1),
            session_start_ist=self.session_start_ist,
            session_end_ist=self.session_end_ist,
            tz_display=self.tz_display,
        )
        
        # Find the session window containing ts_utc
        session_window = None
        for start, end in session_windows:
            if start <= ts_utc < end:
                session_window = (start, end)
                break
        
        if session_window is None:
            # Timestamp outside trading hours
            return
        
        session_start_utc, session_end_utc = session_window
        cache_key = self._get_session_cache_key(ts_utc)
        
        # Check cache
        if cache_key in self._cache:
            return
        
        # Load and build
        ticks_df = self._load_session_data(session_start_utc, session_end_utc)
        session_data = self._build_session_bars(ticks_df, session_start_utc, session_end_utc)
        
        # Cache it
        self._cache[cache_key] = session_data
    
    def iter_timestamps(self, start: datetime, end: datetime) -> Iterator[datetime]:
        """
        Iterate over bar end timestamps in the given range.
        
        Generates timestamps at bar_size frequency, restricted to Mon-Fri sessions.
        
        Args:
            start: Start datetime (timezone-aware, UTC)
            end: End datetime (timezone-aware, UTC)
            
        Yields:
            Bar end timestamps in UTC
        """
        # Generate session windows
        session_windows = generate_session_windows(
            start=start,
            end=end,
            session_start_ist=self.session_start_ist,
            session_end_ist=self.session_end_ist,
            tz_display=self.tz_display,
        )
        
        # For each session, generate bar timestamps
        for session_start_utc, session_end_utc in session_windows:
            bar_timestamps = generate_bar_timestamps(session_start_utc, session_end_utc, self.bar_size)
            
            # Filter to requested range
            mask = (bar_timestamps >= start) & (bar_timestamps <= end)
            session_bars = bar_timestamps[mask]
            
            for bar_ts in session_bars:
                yield bar_ts.to_pydatetime()
    
    def get_snapshot(self, ts: datetime) -> MarketSnapshot:
        """
        Get market snapshot at a specific bar timestamp.
        
        Ensures the session is cached, then retrieves the snapshot for the timestamp.
        
        Args:
            ts: Bar end timestamp in UTC
            
        Returns:
            MarketSnapshot with spot/futures/options data
        """
        # Ensure session is cached
        self._ensure_session_cached(ts)
        
        cache_key = self._get_session_cache_key(ts)
        
        # Get cached data
        if cache_key not in self._cache:
            # Return empty snapshot if session not found
            return MarketSnapshot(timestamp=ts)
        
        session_data = self._cache[cache_key]
        spot_bars = session_data["spot_bars"]
        futures_bars = session_data["futures_bars"]
        option_chain = session_data["option_chain"]
        
        # Get bars for this specific timestamp
        spot_bar = None
        if not spot_bars.empty and ts in spot_bars.index:
            spot_bar = spot_bars.loc[[ts]]  # Return as DataFrame
        
        futures_bar = None
        if not futures_bars.empty and ts in futures_bars.index:
            futures_bar = futures_bars.loc[[ts]]
        
        # Filter option chain to this timestamp
        chain_at_ts = None
        if not option_chain.empty:
            chain_at_ts = option_chain[option_chain["timestamp"] == ts].copy()
            if chain_at_ts.empty:
                chain_at_ts = None
        
        return MarketSnapshot(
            timestamp=ts,
            spot_bar=spot_bar,
            futures_bar=futures_bar,
            options_chain=chain_at_ts,
        )

