"""
Execution model: simulate fills based on LTP ("last") with slippage and fees.

IMPORTANT:
- Your SQLite database does not provide usable bid/ask (they are always 0).
- Therefore execution is modeled as LTP-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from ..config.schemas import ExecutionConfig

logger = logging.getLogger(__name__)


FillOn = Literal["ltp", "bidask", "mid"]  # keep legacy values for backward compatibility
FallbackPrice = Literal["mid", "ltp"]  # kept for backward compatibility (ignored for LTP-only execution)
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


@dataclass
class SimpleBidAskExecution:
    cfg: ExecutionConfig

    def quote(self, chain: Optional[pd.DataFrame], symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return (bid, ask, mid, ltp). Note: bid/ask/mid are not used (LTP-only)."""
        row = _row_for_symbol(chain, symbol)
        if row is None:
            return None, None, None, None
        ltp = row.get("last") if "last" in row else row.get("ltp")
        try:
            ltp_f = float(ltp) if ltp is not None else None
        except Exception:
            ltp_f = None
        return None, None, None, ltp_f

    def fill(self, chain: Optional[pd.DataFrame], symbol: str, side: Side, qty: int) -> Optional[Fill]:
        """
        Simulate a fill for a symbol at current bar snapshot.
        
        LTP-only execution:
        - BUY fill = LTP * (1 + slippage_bps/10000)
        - SELL fill = LTP * (1 - slippage_bps/10000)
        """
        _, _, _, ltp = self.quote(chain, symbol)
        slip = float(self.cfg.slippage_bps) / 10000.0

        if ltp is None or not np.isfinite(ltp) or ltp <= 0:
            return None

        if side == "BUY":
            px = float(ltp) * (1.0 + slip)
        else:
            px = float(ltp) * (1.0 - slip)
        fill_mode = "ltp_with_slippage"

        if px is None or not np.isfinite(px) or px <= 0:
            return None

        logger.info(f"fill_mode={fill_mode} symbol={symbol} side={side} qty={int(qty)} ltp={ltp} slip_bps={self.cfg.slippage_bps}")

        commission = float(self.cfg.commission_per_contract) * int(qty)
        fees = float(self.cfg.fees_per_contract) * int(qty)
        return Fill(symbol=symbol, side=side, qty=int(qty), price=float(px), commission=commission, fees=fees)

    def fill_on_next_bar_open(
        self,
        *,
        chain: Optional[pd.DataFrame],
        options_ticks: Optional[pd.DataFrame],
        symbol: str,
        side: Side,
        qty: int,
        bar_start_utc: datetime,
        bar_end_utc: datetime,
    ) -> Optional[Fill]:
        """
        Realistic bar-based fill: use the first option tick inside the next bar as "open",
        then apply slippage.

        Window policy: (bar_start_utc, bar_end_utc] (i.e., first tick strictly after bar_start)
        If no tick is available for the symbol in the bar window, fall back to the snapshot LTP
        (chain "last") at bar_end_utc.
        """
        qty = int(qty)
        if qty <= 0:
            return None

        slip = float(self.cfg.slippage_bps) / 10000.0

        open_ltp: Optional[float] = None
        fill_mode = "bar_open_tick_with_slippage"

        if options_ticks is not None and not options_ticks.empty:
            df = options_ticks
            if "ts" in df.columns and "symbol" in df.columns and ("ltp" in df.columns or "last" in df.columns):
                ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
                s = df["symbol"].astype(str)
                px_col = "ltp" if "ltp" in df.columns else "last"
                px = pd.to_numeric(df[px_col], errors="coerce")
                mask = (s == str(symbol)) & (ts > pd.Timestamp(bar_start_utc)) & (ts <= pd.Timestamp(bar_end_utc))
                w = df.loc[mask].copy()
                if not w.empty:
                    # ensure chronological by ts (provider loads ordered, but be defensive)
                    w_ts = pd.to_datetime(w["ts"], utc=True, errors="coerce")
                    w_px = pd.to_numeric(w[px_col], errors="coerce")
                    w = w.assign(_ts=w_ts, _px=w_px).sort_values("_ts", ascending=True)
                    v = w["_px"].iloc[0]
                    try:
                        open_ltp = float(v) if v is not None else None
                    except Exception:
                        open_ltp = None

        if open_ltp is None or not np.isfinite(open_ltp) or open_ltp <= 0:
            # fall back to snapshot LTP at bar end
            _, _, _, ltp = self.quote(chain, symbol)
            if ltp is None or not np.isfinite(ltp) or ltp <= 0:
                return None
            open_ltp = float(ltp)
            fill_mode = "bar_open_missing_fallback_to_snapshot_ltp_with_slippage"

        if side == "BUY":
            px_fill = float(open_ltp) * (1.0 + slip)
        else:
            px_fill = float(open_ltp) * (1.0 - slip)

        if px_fill is None or not np.isfinite(px_fill) or px_fill <= 0:
            return None

        logger.info(
            f"fill_mode={fill_mode} symbol={symbol} side={side} qty={int(qty)} "
            f"open_ltp={open_ltp} bar_start={bar_start_utc.isoformat()} bar_end={bar_end_utc.isoformat()} slip_bps={self.cfg.slippage_bps}"
        )

        commission = float(self.cfg.commission_per_contract) * int(qty)
        fees = float(self.cfg.fees_per_contract) * int(qty)
        return Fill(symbol=symbol, side=side, qty=int(qty), price=float(px_fill), commission=commission, fees=fees)


