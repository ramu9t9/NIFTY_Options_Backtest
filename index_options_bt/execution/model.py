"""
Execution model: simulate fills based on bid/ask with slippage and fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.schemas import ExecutionConfig


FillOn = Literal["bidask", "mid"]
FallbackPrice = Literal["mid", "ltp"]
Side = Literal["BUY", "SELL"]


@dataclass
class Fill:
    symbol: str
    side: Side
    qty: int
    price: float
    commission: float
    fees: float

    @property
    def total_cost(self) -> float:
        # BUY consumes cash; SELL adds cash (handled by portfolio)
        return self.commission + self.fees


def _row_for_symbol(chain: Optional[pd.DataFrame], symbol: str) -> Optional[pd.Series]:
    if chain is None or chain.empty:
        return None
    df = chain[chain["symbol"] == symbol]
    if df.empty:
        return None
    return df.iloc[0]


def _mid(row: pd.Series) -> Optional[float]:
    """
    Calculate mid price from bid/ask.
    Returns None if bid/ask are missing or both are 0 (use LTP instead).
    """
    bid = row.get("bid")
    ask = row.get("ask")
    if bid is None or ask is None:
        return None
    try:
        bid_f = float(bid)
        ask_f = float(ask)
    except Exception:
        return None
    if not np.isfinite(bid_f) or not np.isfinite(ask_f):
        return None
    # If both bid and ask are 0, return None (use LTP instead, like old engine)
    if bid_f == 0.0 and ask_f == 0.0:
        return None
    return (bid_f + ask_f) / 2.0


@dataclass
class SimpleBidAskExecution:
    cfg: ExecutionConfig

    def quote(self, chain: Optional[pd.DataFrame], symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (bid, ask, mid, ltp) from chain row."""
        row = _row_for_symbol(chain, symbol)
        if row is None:
            return None, None, None, None
        bid = row.get("bid")
        ask = row.get("ask")
        ltp = row.get("last") if "last" in row else row.get("ltp")
        try:
            bid_f = float(bid) if bid is not None else None
        except Exception:
            bid_f = None
        try:
            ask_f = float(ask) if ask is not None else None
        except Exception:
            ask_f = None
        try:
            ltp_f = float(ltp) if ltp is not None else None
        except Exception:
            ltp_f = None
        mid_f = _mid(row)
        return bid_f, ask_f, mid_f, ltp_f

    def fill(self, chain: Optional[pd.DataFrame], symbol: str, side: Side, qty: int) -> Optional[Fill]:
        """
        Simulate a fill for a symbol at current bar snapshot.
        
        IMPORTANT: Like the old backtest engine, we prioritize LTP when bid/ask are 0 or missing.
        This matches real-world behavior where many options trade on LTP.
        """
        bid, ask, mid, ltp = self.quote(chain, symbol)
        slip = float(self.cfg.slippage_bps) / 10000.0

        px: Optional[float] = None
        
        # Try bid/ask first if available and non-zero
        if self.cfg.fill_on == "bidask":
            if side == "BUY":
                if ask is not None and np.isfinite(ask) and ask > 0:
                    px = float(ask) * (1.0 + slip)
            else:
                if bid is not None and np.isfinite(bid) and bid > 0:
                    px = float(bid) * (1.0 - slip)

        # Fallback: prioritize LTP (like old engine) when bid/ask are missing or 0
        if px is None:
            if ltp is not None and np.isfinite(ltp) and ltp > 0:
                # Use LTP directly (old engine behavior)
                px = float(ltp)
            elif mid is not None and np.isfinite(mid) and mid > 0:
                # Use mid as last resort
                px = float(mid)

        if px is None or not np.isfinite(px) or px <= 0:
            return None

        commission = float(self.cfg.commission_per_contract) * int(qty)
        fees = float(self.cfg.fees_per_contract) * int(qty)
        return Fill(symbol=symbol, side=side, qty=int(qty), price=float(px), commission=commission, fees=fees)


