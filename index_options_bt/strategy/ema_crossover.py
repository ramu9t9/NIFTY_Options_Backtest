"""
EMA Crossover Strategy.

Simple moving average crossover strategy:
- Fast EMA crosses above Slow EMA -> LONG CALL
- Fast EMA crosses below Slow EMA -> LONG PUT
- Exit on opposite signal.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import Strategy, Intent
from ..data.models import MarketSnapshot
from .registry import register_strategy


@register_strategy("ema_crossover")
class EMACrossoverStrategy(Strategy):
    """
    Simple EMA crossover strategy based on spot price movement.
    
    Parameters:
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        min_bars: Minimum bars required before generating signals (default: slow_period)
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.fast_period = params.get("fast_period", 12)
        self.slow_period = params.get("slow_period", 26)
        self.min_bars = params.get("min_bars", self.slow_period)
        self.stop_loss_pct = params.get("stop_loss_pct", None)
        self.take_profit_pct = params.get("take_profit_pct", None)
        
        # Internal state
        self._spot_prices: List[float] = []
        self._fast_ema: List[float] = []
        self._slow_ema: List[float] = []
        self._last_signal: Optional[str] = None  # "LONG", "SHORT", or None
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None
        
        # Get the appropriate EMA list
        ema_list = self._fast_ema if period == self.fast_period else self._slow_ema
        
        # If we have previous EMA value, use it for calculation
        if ema_list:
            prev_ema = ema_list[-1]
            k = 2.0 / (period + 1)
            current_price = prices[-1]
            ema = (current_price * k) + (prev_ema * (1 - k))
            return ema
        else:
            # First EMA value: use SMA
            return sum(prices[-period:]) / period
    
    def generate_intents(
        self,
        snapshot: MarketSnapshot,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Intent]:
        """
        Generate EMA crossover intents based on spot price movement.
        
        Strategy logic:
        1. Track spot price history
        2. Calculate Fast EMA and Slow EMA
        3. Detect crossover:
           - Fast EMA > Slow EMA (golden cross) -> LONG CALL
           - Fast EMA < Slow EMA (death cross) -> LONG PUT
        4. Exit on opposite signal (handled by execution layer)
        """
        if not snapshot.has_spot():
            return []
        
        spot_price = snapshot.get_spot_price()
        if spot_price is None:
            return []
        
        # Update price history
        self._spot_prices.append(spot_price)
        
        # Need enough data to calculate EMAs
        if len(self._spot_prices) < self.min_bars:
            return []
        
        # Calculate Fast EMA
        fast_ema = self._calculate_ema(self._spot_prices, self.fast_period)
        if fast_ema is None:
            return []
        self._fast_ema.append(fast_ema)
        
        # Calculate Slow EMA
        slow_ema = self._calculate_ema(self._spot_prices, self.slow_period)
        if slow_ema is None:
            return []
        self._slow_ema.append(slow_ema)
        
        # Keep only necessary history for performance
        if len(self._spot_prices) > self.slow_period * 2:
            self._spot_prices = self._spot_prices[-self.slow_period * 2:]
            if len(self._fast_ema) > self.slow_period:
                self._fast_ema = self._fast_ema[-self.slow_period:]
            if len(self._slow_ema) > self.slow_period:
                self._slow_ema = self._slow_ema[-self.slow_period:]
        
        # Need at least 2 EMA values to detect crossover
        if len(self._fast_ema) < 2 or len(self._slow_ema) < 2:
            return []
        
        # Detect crossover
        prev_fast = self._fast_ema[-2]
        prev_slow = self._slow_ema[-2]
        curr_fast = self._fast_ema[-1]
        curr_slow = self._slow_ema[-1]
        
        # Golden cross: Fast EMA crosses above Slow EMA
        golden_cross = (prev_fast <= prev_slow) and (curr_fast > curr_slow)
        
        # Death cross: Fast EMA crosses below Slow EMA
        death_cross = (prev_fast >= prev_slow) and (curr_fast < curr_slow)
        
        if golden_cross:
            # Generate LONG CALL intent
            if self._last_signal != "LONG":
                prev_signal = self._last_signal
                metadata = {
                    "reason": "golden_cross",
                    "fast_ema": fast_ema,
                    "slow_ema": slow_ema,
                    "spot_price": spot_price,
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                }
                if self.stop_loss_pct is not None:
                    metadata["stop_loss_pct"] = self.stop_loss_pct
                if self.take_profit_pct is not None:
                    metadata["take_profit_pct"] = self.take_profit_pct
                
                intent = Intent(
                    timestamp=snapshot.timestamp,
                    direction="LONG",
                    option_type="CALL",
                    size=1,
                    metadata=metadata,
                )
                self._last_signal = "LONG"
                # If we were previously SHORT, emit a FLAT first to close existing positions, then enter.
                if prev_signal == "SHORT":
                    return [Intent(timestamp=snapshot.timestamp, direction="FLAT", metadata={"reason": "opposite_signal"}), intent]
                return [intent]
        
        elif death_cross:
            # Generate LONG PUT intent
            if self._last_signal != "SHORT":
                prev_signal = self._last_signal
                metadata = {
                    "reason": "death_cross",
                    "fast_ema": fast_ema,
                    "slow_ema": slow_ema,
                    "spot_price": spot_price,
                    "fast_period": self.fast_period,
                    "slow_period": self.slow_period,
                }
                if self.stop_loss_pct is not None:
                    metadata["stop_loss_pct"] = self.stop_loss_pct
                if self.take_profit_pct is not None:
                    metadata["take_profit_pct"] = self.take_profit_pct
                
                intent = Intent(
                    timestamp=snapshot.timestamp,
                    direction="LONG",
                    option_type="PUT",
                    size=1,
                    metadata=metadata,
                )
                self._last_signal = "SHORT"
                # If we were previously LONG, emit a FLAT first to close existing positions, then enter.
                if prev_signal == "LONG":
                    return [Intent(timestamp=snapshot.timestamp, direction="FLAT", metadata={"reason": "opposite_signal"}), intent]
                return [intent]
        
        # No crossover, but check if we should exit (optional: exit on opposite trend)
        # For simplicity, we'll let the execution layer handle exits via stop-loss
        # or opposite signals will be generated on next crossover
        
        return []

