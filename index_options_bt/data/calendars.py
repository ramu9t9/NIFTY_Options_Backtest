"""
Market session calendar utilities for NIFTY trading hours.

Generates Mon-Fri IST sessions (09:15-15:30) and converts to UTC for database queries.
"""

from datetime import datetime, time, timedelta, timezone
from typing import List, Tuple
import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc

# Default market hours (IST)
DEFAULT_SESSION_START_IST = time(9, 15)  # 09:15 IST
DEFAULT_SESSION_END_IST = time(15, 30)  # 15:30 IST


def generate_session_windows(
    start: datetime,
    end: datetime,
    session_start_ist: time | str = DEFAULT_SESSION_START_IST,
    session_end_ist: time | str = DEFAULT_SESSION_END_IST,
    tz_display: timezone = IST,
) -> List[Tuple[datetime, datetime]]:
    """
    Generate trading session windows between start and end dates.
    
    Sessions are Mon-Fri only (no holiday calendar for now).
    Each window is returned as (session_start_utc, session_end_utc).
    
    Args:
        start: Start datetime (timezone-aware, will be normalized to IST for date extraction)
        end: End datetime (timezone-aware)
        session_start_ist: Market session start time in IST (default 09:15) - can be time object or "HH:MM" string
        session_end_ist: Market session end time in IST (default 15:30) - can be time object or "HH:MM" string
        tz_display: Display timezone (IST) for date extraction
        
    Returns:
        List of (session_start_utc, session_end_utc) tuples
        
    Example:
        >>> start = datetime(2025, 10, 1, 9, 0, tzinfo=UTC)
        >>> end = datetime(2025, 10, 3, 16, 0, tzinfo=UTC)
        >>> windows = generate_session_windows(start, end)
        >>> len(windows)  # Mon, Tue, Wed sessions
        3
    """
    # Parse session times if strings
    if isinstance(session_start_ist, str):
        hour, minute = map(int, session_start_ist.split(":"))
        session_start_ist = time(hour, minute)
    if isinstance(session_end_ist, str):
        hour, minute = map(int, session_end_ist.split(":"))
        session_end_ist = time(hour, minute)
    
    # Normalize start/end to IST for date extraction
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    
    start_ist = start.astimezone(tz_display)
    end_ist = end.astimezone(tz_display)
    
    windows = []
    current_date = start_ist.date()
    end_date = end_ist.date()
    
    while current_date <= end_date:
        # Skip weekends (Mon=0, Fri=4)
        weekday = current_date.weekday()
        if weekday < 5:  # Mon-Fri
            session_start_dt = datetime.combine(current_date, session_start_ist).replace(tzinfo=tz_display)
            session_end_dt = datetime.combine(current_date, session_end_ist).replace(tzinfo=tz_display)
            
            # Convert to UTC
            session_start_utc = session_start_dt.astimezone(UTC)
            session_end_utc = session_end_dt.astimezone(UTC)
            
            # Only include sessions that overlap with the requested range
            if session_end_utc >= start and session_start_utc <= end:
                windows.append((session_start_utc, session_end_utc))
        
        current_date += timedelta(days=1)
    
    return windows


def generate_bar_timestamps(
    session_start_utc: datetime,
    session_end_utc: datetime,
    bar_size: str,
) -> pd.DatetimeIndex:
    """
    Generate bar end timestamps for a trading session at the specified bar size.
    
    Bar ends are aligned to the session start (e.g., if session starts at 09:15:00 IST
    and bar_size is "15s", bars end at 09:15:15, 09:15:30, 09:15:45, etc.).
    
    Args:
        session_start_utc: Session start in UTC
        session_end_utc: Session end in UTC
        bar_size: Pandas frequency string (e.g., "5s", "15s", "30s", "1min")
        
    Returns:
        DatetimeIndex of bar end timestamps in UTC
        
    Example:
        >>> start = datetime(2025, 10, 1, 3, 45, tzinfo=UTC)  # 09:15 IST
        >>> end = datetime(2025, 10, 1, 10, 0, tzinfo=UTC)  # 15:30 IST
        >>> bars = generate_bar_timestamps(start, end, "15s")
        >>> len(bars)  # ~22800 bars for 6.25 hour session at 15s
        >>> bars[0]  # First bar end
        Timestamp('2025-10-01 03:45:15+00:00', tz='UTC')
    """
    # Generate range from session_start to session_end at bar_size frequency
    # Bar ends start after the first bar_size interval
    bar_range = pd.date_range(
        start=session_start_utc,
        end=session_end_utc,
        freq=bar_size,
        inclusive="right",  # Include end, exclude start initially
    )
    
    # Actually, we want bar ends starting from session_start + bar_size
    # So shift to include the first bar end
    if len(bar_range) == 0:
        # Edge case: session too short
        return pd.DatetimeIndex([], tz=UTC)
    
    # Ensure we capture all bars by starting from session_start and including session_end
    full_range = pd.date_range(
        start=session_start_utc,
        end=session_end_utc,
        freq=bar_size,
    )
    
    # Filter to only bars that end within the session
    # A bar ending at T means it contains data from [T-bar_size, T)
    # So bar ends should be > session_start and <= session_end
    mask = (full_range > session_start_utc) & (full_range <= session_end_utc)
    return full_range[mask]

