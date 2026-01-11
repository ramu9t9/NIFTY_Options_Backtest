"""
Example Breakout Strategy.

Spot breakout -> buy ATM call (or put) with nearest weekly expiry.
Exit on opposite signal or stop-loss on option premium.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from .base import Strategy, Intent
from ..data.models import MarketSnapshot

from .registry import register_strategy


@register_strategy("breakout")
class ExampleBreakoutStrategy(Strategy):
    """
    Simple breakout strategy based on spot price movement.
    
    Parameters:
        lookback_bars: Number of bars to look back for breakout detection (default: 20)
        stop_loss_pct: Stop loss percentage on option premium (default: 0.35)
        breakout_threshold_pct: Percentage move to trigger breakout (default: 0.11)
    """
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.lookback_bars = params.get("lookback_bars", 20)
        self.stop_loss_pct = params.get("stop_loss_pct", 0.35)
        self.take_profit_pct = params.get("take_profit_pct", None)
        self.breakout_threshold_pct = params.get("breakout_threshold_pct", 0.11)
        
        # Internal state
        self._spot_history: List[float] = []
        self._last_direction: Optional[str] = None
    
    def generate_intents(
        self,
        snapshot: MarketSnapshot,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Intent]:
        """
        Generate breakout intents based on spot price movement.
        
        Strategy logic:
        1. Track spot price history
        2. Detect breakout: if spot moves > breakout_threshold_pct from recent low/high
        3. Generate LONG CALL on upward breakout, LONG PUT on downward breakout
        4. Exit on opposite signal or stop-loss (handled by execution layer)
        """
        if not snapshot.has_spot():
            return []
        
        spot_price = snapshot.get_spot_price()
        if spot_price is None:
            return []
        
        # Update history
        self._spot_history.append(spot_price)
        
        # Keep enough history to compute recent high/low *excluding* current bar.
        # Need lookback_bars prior bars + current bar => lookback_bars + 1 total.
        max_len = self.lookback_bars + 1
        if len(self._spot_history) > max_len:
            self._spot_history = self._spot_history[-max_len:]
        
        # Need enough history to detect breakout (prior lookback bars + current bar)
        if len(self._spot_history) < (self.lookback_bars + 1):
            return []
        
        # Calculate recent high/low over PRIOR bars only (exclude current bar)
        prior_window = self._spot_history[-(self.lookback_bars + 1) : -1]
        recent_high = max(prior_window)
        recent_low = min(prior_window)

        current_price = self._spot_history[-1]
        breakout_direction = None
        
        # Upward breakout: price breaks above recent high
        if current_price >= recent_high * (1 + self.breakout_threshold_pct / 100):
            breakout_direction = "UP"
        
        # Downward breakout: price breaks below recent low
        elif current_price <= recent_low * (1 - self.breakout_threshold_pct / 100):
            breakout_direction = "DOWN"
        
        if breakout_direction is None:
            # No breakout, return FLAT or no intent (depends on current position)
            # For simplicity, return FLAT if we had a position
            if self._last_direction is not None:
                intent = Intent(
                    timestamp=snapshot.timestamp,
                    direction="FLAT",
                    metadata={"reason": "no_breakout", "spot_price": spot_price},
                )
                self._last_direction = None
                return [intent]
            return []
        
        # Generate intent based on breakout direction
        if breakout_direction == "UP":
            option_type = "CALL"
            direction = "LONG"
        else:  # DOWN
            option_type = "PUT"
            direction = "LONG"
        
        # Only generate new intent if direction changed
        if self._last_direction == breakout_direction:
            return []
        
        self._last_direction = breakout_direction
        
        metadata = {
            "reason": "breakout",
            "breakout_direction": breakout_direction,
            "spot_price": spot_price,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "lookback_bars": self.lookback_bars,
        }
        if self.stop_loss_pct is not None:
            metadata["stop_loss_pct"] = self.stop_loss_pct
        if self.take_profit_pct is not None:
            metadata["take_profit_pct"] = self.take_profit_pct
        
        intent = Intent(
            timestamp=snapshot.timestamp,
            direction=direction,
            option_type=option_type,
            size=1,
            metadata=metadata,
        )
        
        return [intent]

