"""
Data models for market snapshots and provider interfaces.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol, Iterator
import pandas as pd


@dataclass
class MarketSnapshot:
    """
    Market snapshot at a specific timestamp.
    
    Contains:
    - Spot/index bar (OHLCV)
    - Futures bar (optional)
    - Options chain snapshot DataFrame
    """
    timestamp: datetime  # Bar end timestamp in UTC
    spot_bar: Optional[pd.DataFrame] = None  # OHLCV bar for index (columns: open, high, low, close, volume, oi)
    futures_bar: Optional[pd.DataFrame] = None  # OHLCV bar for futures (optional)
    options_chain: Optional[pd.DataFrame] = None  # Chain snapshot with columns: timestamp, symbol, expiry, strike, cp, bid, ask, last, iv, delta, gamma, theta, vega, volume, oi
    
    def has_spot(self) -> bool:
        """Check if spot bar is available"""
        return self.spot_bar is not None and not self.spot_bar.empty
    
    def has_futures(self) -> bool:
        """Check if futures bar is available"""
        return self.futures_bar is not None and not self.futures_bar.empty
    
    def has_options(self) -> bool:
        """Check if options chain is available"""
        return self.options_chain is not None and not self.options_chain.empty
    
    def get_spot_price(self) -> Optional[float]:
        """Get spot close price"""
        if not self.has_spot():
            return None
        return float(self.spot_bar.iloc[0]["close"])


class DataProvider(Protocol):
    """
    Protocol for data providers.
    
    Providers yield market snapshots at bar timestamps.
    """
    
    def iter_timestamps(self, start: datetime, end: datetime) -> Iterator[datetime]:
        """
        Iterate over bar end timestamps in the given range.
        
        Args:
            start: Start datetime (timezone-aware, UTC)
            end: End datetime (timezone-aware, UTC)
            
        Yields:
            Bar end timestamps in UTC
        """
        ...
    
    def get_snapshot(self, ts: datetime) -> MarketSnapshot:
        """
        Get market snapshot at a specific bar timestamp.
        
        Args:
            ts: Bar end timestamp in UTC
            
        Returns:
            MarketSnapshot with spot/futures/options data
        """
        ...

