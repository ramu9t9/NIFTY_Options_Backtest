"""
Risk management: sizing + pre-trade checks.

This module is intentionally conservative and "fail-closed":
- If a limit is configured and we cannot estimate required values, we reduce size or reject.
- If no limits are configured, behavior matches the existing engine defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..config.schemas import RiskConfig


def _pct_to_frac(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    v = float(x)
    # Accept both 0.10 and 10.0 forms
    if v > 1.0:
        v = v / 100.0
    return v


@dataclass
class RiskDecision:
    approved_qty: int
    reason: str = "ok"


@dataclass
class RiskManager:
    cfg: RiskConfig

    def cap_qty_for_trade(
        self,
        *,
        desired_qty: int,
        price: Optional[float],
        multiplier: int,
        cash_available: float,
        portfolio_abs_notional: Optional[float],
        stop_loss_pct: Optional[float],
        realized_pnl_today: Optional[float] = None,
    ) -> RiskDecision:
        """
        Decide final qty for a new entry.

        Args:
            desired_qty: desired contracts
            price: option premium estimate (LTP). If None, we may reject when limits require it.
            multiplier: lot size
            cash_available: current cash
            portfolio_abs_notional: current absolute notional exposure (approx). None if unknown.
            stop_loss_pct: strategy stop-loss percent (or fraction). Used to estimate max loss per trade.
            realized_pnl_today: realized PnL for current day (INR). Used for max_loss_per_day.
        """
        qty = int(max(0, desired_qty))
        if qty == 0:
            return RiskDecision(approved_qty=0, reason="qty=0")

        mult = int(max(1, multiplier))
        px = float(price) if price is not None else None

        # Max contracts
        if self.cfg.max_contracts is not None:
            qty = min(qty, int(self.cfg.max_contracts))

        # If any money-based limit is set, we need a price.
        needs_price = any(
            x is not None
            for x in [self.cfg.max_notional, self.cfg.max_loss_per_trade, self.cfg.margin_per_contract]
        )
        if needs_price and (px is None or px <= 0):
            return RiskDecision(approved_qty=0, reason="missing_price_for_limits")

        # Margin / cash sanity: for BUY we pay premium; for simplicity treat margin_per_contract as extra reserve.
        if px is not None and px > 0:
            premium_cost_per_contract = px * mult
            reserve_per_contract = float(self.cfg.margin_per_contract or 0.0)
            max_affordable = int(cash_available // (premium_cost_per_contract + reserve_per_contract)) if (premium_cost_per_contract + reserve_per_contract) > 0 else qty
            qty = min(qty, max_affordable)
            if qty <= 0:
                return RiskDecision(approved_qty=0, reason="insufficient_cash")

        # Max notional (approx absolute exposure)
        if self.cfg.max_notional is not None and px is not None and px > 0:
            current = float(portfolio_abs_notional or 0.0)
            per_contract = px * mult
            remaining = float(self.cfg.max_notional) - current
            if remaining <= 0:
                return RiskDecision(approved_qty=0, reason="max_notional_reached")
            max_by_notional = int(remaining // per_contract) if per_contract > 0 else qty
            qty = min(qty, max_by_notional)
            if qty <= 0:
                return RiskDecision(approved_qty=0, reason="max_notional_reached")

        # Max loss per trade: estimate using stop loss if provided, else assume full premium at risk.
        if self.cfg.max_loss_per_trade is not None and px is not None and px > 0:
            sl = _pct_to_frac(stop_loss_pct)
            loss_frac = float(sl) if sl is not None else 1.0
            est_loss_per_contract = px * mult * loss_frac
            if est_loss_per_contract <= 0:
                return RiskDecision(approved_qty=0, reason="bad_loss_estimate")
            max_by_loss = int(float(self.cfg.max_loss_per_trade) // est_loss_per_contract)
            qty = min(qty, max_by_loss)
            if qty <= 0:
                return RiskDecision(approved_qty=0, reason="max_loss_per_trade")

        # Max loss per day: if already below threshold, block new trades.
        if self.cfg.max_loss_per_day is not None and realized_pnl_today is not None:
            if float(realized_pnl_today) <= -abs(float(self.cfg.max_loss_per_day)):
                return RiskDecision(approved_qty=0, reason="max_loss_per_day")

        return RiskDecision(approved_qty=int(qty), reason="ok")


