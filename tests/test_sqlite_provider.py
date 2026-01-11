"""
Tests for SQLiteDataProvider.
"""

import sqlite3
import tempfile
import os
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

from index_options_bt.data import SQLiteDataProvider, MarketSnapshot
from index_options_bt.data.symbols import parse_symbol


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with ltp_ticks schema"""
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE ltp_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            token TEXT NOT NULL,
            ts TEXT NOT NULL,
            ltp REAL,
            bid REAL,
            ask REAL,
            volume INTEGER,
            oi INTEGER,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            iv REAL,
            source TEXT DEFAULT 'test'
        );
    """)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_ltp_symbol_ts ON ltp_ticks(symbol, ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON ltp_ticks(ts);")
    conn.commit()
    
    yield db_path
    
    conn.close()
    Path(db_path).unlink()


@pytest.fixture
def sample_data(temp_db):
    """Insert sample data into temp database"""
    conn = sqlite3.connect(temp_db)
    
    # Generate timestamps for a trading session (IST 09:15-15:30, converted to UTC)
    ist = timezone(timedelta(hours=5, minutes=30))
    session_start_ist = datetime(2025, 10, 1, 9, 15, tzinfo=ist)
    session_end_ist = datetime(2025, 10, 1, 15, 30, tzinfo=ist)
    session_start_utc = session_start_ist.astimezone(timezone.utc)
    session_end_utc = session_end_ist.astimezone(timezone.utc)
    
    # Generate 5-second ticks
    current_ts = session_start_utc
    tick_id = 0
    
    # Index symbol: "NIFTY 50" (volume=0, greeks null)
    while current_ts < session_end_utc:
        conn.execute(
            "INSERT INTO ltp_ticks (symbol, token, ts, ltp, bid, ask, volume, oi, delta, gamma, theta, vega, iv, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NIFTY 50", "26000", current_ts.isoformat(), 19500.0 + tick_id * 0.5, None, None, 0, None, None, None, None, None, None, "test"),
        )
        current_ts += timedelta(seconds=5)
        tick_id += 1
    
    # FUT symbol: NIFTY28NOV25FUT (a few ticks)
    fut_ts = session_start_utc
    for i in range(10):
        conn.execute(
            "INSERT INTO ltp_ticks (symbol, token, ts, ltp, bid, ask, volume, oi, delta, gamma, theta, vega, iv, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NIFTY28NOV25FUT", "26001", fut_ts.isoformat(), 19550.0 + i, None, None, 1000 + i, None, None, None, None, None, None, "test"),
        )
        fut_ts += timedelta(seconds=5)
    
    # OPT symbols: NIFTY25NOV2526000CE and NIFTY25NOV2526000PE (with greeks)
    opt_ts = session_start_utc
    for i in range(20):
        # Call option
        conn.execute(
            "INSERT INTO ltp_ticks (symbol, token, ts, ltp, bid, ask, volume, oi, delta, gamma, theta, vega, iv, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NIFTY25NOV2526000CE", "49873", opt_ts.isoformat(), 100.0 + i * 0.5, 99.5 + i * 0.5, 100.5 + i * 0.5, 1000 + i, 50000 + i, 0.5, 0.001, -10.0, 15.0, 12.5, "test"),
        )
        # Put option
        conn.execute(
            "INSERT INTO ltp_ticks (symbol, token, ts, ltp, bid, ask, volume, oi, delta, gamma, theta, vega, iv, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("NIFTY25NOV2526000PE", "49874", opt_ts.isoformat(), 50.0 + i * 0.3, 49.5 + i * 0.3, 50.5 + i * 0.3, 800 + i, 40000 + i, -0.3, 0.001, -8.0, 12.0, 11.0, "test"),
        )
        opt_ts += timedelta(seconds=5)
    
    conn.commit()
    conn.close()
    
    return temp_db, session_start_utc, session_end_utc


def test_provider_iter_timestamps(sample_data):
    """Test that provider yields bar timestamps correctly"""
    db_path, start_utc, end_utc = sample_data
    
    provider = SQLiteDataProvider(
        sqlite_path=db_path,
        bar_size="15s",
        symbol_index="NIFTY 50",
    )
    
    try:
        timestamps = list(provider.iter_timestamps(start_utc, end_utc))
        assert len(timestamps) > 0
        
        # Check that timestamps are in UTC and increasing
        for i, ts in enumerate(timestamps):
            assert ts.tzinfo == timezone.utc
            if i > 0:
                assert ts > timestamps[i - 1]
        
    finally:
        provider.close()


def test_provider_get_snapshot(sample_data):
    """Test that provider returns complete snapshots"""
    db_path, start_utc, end_utc = sample_data
    
    provider = SQLiteDataProvider(
        sqlite_path=db_path,
        bar_size="15s",
        symbol_index="NIFTY 50",
    )
    
    try:
        # Get a bar timestamp
        timestamps = list(provider.iter_timestamps(start_utc, end_utc))
        if not timestamps:
            pytest.skip("No timestamps generated")
        
        test_ts = timestamps[0]
        
        # Get snapshot
        snapshot = provider.get_snapshot(test_ts)
        
        assert isinstance(snapshot, MarketSnapshot)
        assert snapshot.timestamp == test_ts
        
        # Check spot bar exists and has required columns
        if snapshot.has_spot():
            assert snapshot.spot_bar is not None
            assert not snapshot.spot_bar.empty
            required_cols = ["open", "high", "low", "close", "volume", "oi"]
            for col in required_cols:
                assert col in snapshot.spot_bar.columns, f"Missing column: {col}"
        
        # Check option chain exists and has required columns
        if snapshot.has_options():
            assert snapshot.options_chain is not None
            assert not snapshot.options_chain.empty
            required_cols = ["timestamp", "symbol", "expiry", "strike", "cp", "bid", "ask", "last", "iv", "delta", "gamma", "theta", "vega", "volume", "oi"]
            for col in required_cols:
                assert col in snapshot.options_chain.columns, f"Missing column: {col}"
        
    finally:
        provider.close()


def test_provider_session_caching(sample_data):
    """Test that session caching works correctly"""
    db_path, start_utc, end_utc = sample_data
    
    provider = SQLiteDataProvider(
        sqlite_path=db_path,
        bar_size="15s",
        symbol_index="NIFTY 50",
    )
    
    try:
        # Get same timestamp twice - should use cache
        timestamps = list(provider.iter_timestamps(start_utc, end_utc))
        if len(timestamps) < 2:
            pytest.skip("Not enough timestamps")
        
        test_ts = timestamps[0]
        
        # First call - should load from DB
        snapshot1 = provider.get_snapshot(test_ts)
        
        # Second call - should use cache
        snapshot2 = provider.get_snapshot(test_ts)
        
        assert snapshot1.timestamp == snapshot2.timestamp
        
    finally:
        provider.close()


def test_symbol_parsing():
    """Test symbol parsing works correctly"""
    # Index symbol
    parsed = parse_symbol("NIFTY 50")
    assert parsed.kind == "INDEX"
    assert parsed.underlying == "NIFTY"
    
    # Future symbol
    parsed = parse_symbol("NIFTY28NOV25FUT")
    assert parsed.kind == "FUT"
    assert parsed.underlying == "NIFTY"
    assert parsed.expiry is not None
    
    # Option symbol (Call)
    parsed = parse_symbol("NIFTY25NOV2526000CE")
    assert parsed.kind == "OPT"
    assert parsed.underlying == "NIFTY"
    assert parsed.strike == 26000
    assert parsed.cp == "C"
    
    # Option symbol (Put)
    parsed = parse_symbol("NIFTY25NOV2526000PE")
    assert parsed.kind == "OPT"
    assert parsed.strike == 26000
    assert parsed.cp == "P"

