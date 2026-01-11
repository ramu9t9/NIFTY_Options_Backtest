"""
Minimal portfolio for options backtesting:
- cash
- positions keyed by contract_id
- realized/unrealized PnL
- MTM from chain mid/ltp
- minimal expiry cash settlement
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, time, timezone, timedelta
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.models import MarketSnapshot
from ..data.symbols import option_contract_id, parse_symbol

Side = Literal["BUY", "SELL"]


@dataclass
class TradeRecord:
    trade_id: int
    timestamp: datetime
    symbol: str
    contract_id: str
    side: Side
    qty: int
    price: float
    commission: float
    fees: float
    realized_pnl_after: float
    cash_after: float


@dataclass
class Position:
    contract_id: str
    symbol: str
    qty: int = 0  # signed
    avg_price: float = 0.0
    multiplier: int = 1
    entry_ts: Optional[datetime] = None
    hold_bars: int = 0
    realized_pnl: float = 0.0
    entry_price: Optional[float] = None

    # Optional per-position risk params
    stop_loss_pct: Optional[float] = None  # premium stop (% drawdown from entry)
    take_profit_pct: Optional[float] = None  # premium target (% gain from entry)
    max_hold_bars: Optional[int] = None


@dataclass
class Portfolio:
    initial_cash: float = 100_000.0
    cash: float = 100_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    trades: List[TradeRecord] = field(default_factory=list)
    closed_trades: List[Dict[str, object]] = field(default_factory=list)
    _trade_id: int = 0

    def reset(self) -> None:
        self.cash = float(self.initial_cash)
        self.positions.clear()
        self.trades.clear()
        self._trade_id = 0

    @property
    def realized_pnl(self) -> float:
        return float(sum(p.realized_pnl for p in self.positions.values()))

    def apply_fill(
        self,
        ts: datetime,
        symbol: str,
        side: Side,
        qty: int,
        price: float,
        commission: float = 0.0,
        fees: float = 0.0,
        multiplier: int = 1,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        max_hold_bars: Optional[int] = None,
    ) -> None:
        """Apply a fill to the portfolio and update positions + realized PnL."""
        qty = int(qty)
        if qty <= 0:
            return

        contract_id = option_contract_id(symbol) if "CE" in symbol or "PE" in symbol else symbol

        signed_qty = qty if side == "BUY" else -qty
        pos = self.positions.get(contract_id)
        if pos is None:
            pos = Position(contract_id=contract_id, symbol=symbol, qty=0, avg_price=0.0, multiplier=int(multiplier))
            self.positions[contract_id] = pos

        # Cash impact
        gross = float(price) * qty * pos.multiplier
        costs = float(commission) + float(fees)
        if side == "BUY":
            self.cash -= gross
            self.cash -= costs
        else:
            self.cash += gross
            self.cash -= costs

        # Realized PnL if reducing/closing
        realized_delta = 0.0
        if pos.qty == 0 or (pos.qty > 0 and signed_qty > 0) or (pos.qty < 0 and signed_qty < 0):
            # increasing same direction
            new_qty = pos.qty + signed_qty
            if new_qty != 0:
                # weighted average price on absolute quantities
                old_abs = abs(pos.qty)
                new_abs = abs(new_qty)
                pos.avg_price = (pos.avg_price * old_abs + float(price) * qty) / new_abs
            pos.qty = new_qty
            if pos.entry_ts is None:
                pos.entry_ts = ts
                pos.entry_price = float(price)
                pos.stop_loss_pct = stop_loss_pct
                pos.take_profit_pct = take_profit_pct
                pos.max_hold_bars = max_hold_bars
        else:
            # reducing / flipping
            close_qty = min(abs(pos.qty), qty)
            # pnl per contract = (sell - buy) * signed direction
            if pos.qty > 0 and side == "SELL":
                realized_delta = (float(price) - pos.avg_price) * close_qty * pos.multiplier
                pos.qty -= close_qty
            elif pos.qty < 0 and side == "BUY":
                realized_delta = (pos.avg_price - float(price)) * close_qty * pos.multiplier
                pos.qty += close_qty

            pos.realized_pnl += realized_delta

            # if over-closed, open residual in opposite direction at this price
            residual = qty - close_qty
            if residual > 0:
                # new position direction equals signed_qty (since we fully offset existing)
                pos.qty = (residual if side == "BUY" else -residual)
                pos.avg_price = float(price)
                pos.entry_ts = ts
                pos.hold_bars = 0
                pos.stop_loss_pct = stop_loss_pct
                pos.take_profit_pct = take_profit_pct
                pos.max_hold_bars = max_hold_bars

            if pos.qty == 0:
                # closed roundtrip
                if pos.entry_ts is not None and pos.entry_price is not None:
                    self.closed_trades.append(
                        {
                            "contract_id": pos.contract_id,
                            "symbol": pos.symbol,
                            "entry_time": pos.entry_ts,
                            "exit_time": ts,
                            "entry_price": float(pos.entry_price),
                            "exit_price": float(price),
                            "qty": close_qty,
                            "realized_pnl": float(realized_delta),
                        }
                    )
                pos.avg_price = 0.0
                pos.entry_ts = None
                pos.entry_price = None
                pos.hold_bars = 0
                pos.stop_loss_pct = None
                pos.take_profit_pct = None
                pos.max_hold_bars = None

        self._trade_id += 1
        self.trades.append(
            TradeRecord(
                trade_id=self._trade_id,
                timestamp=ts,
                symbol=symbol,
                contract_id=contract_id,
                side=side,
                qty=qty,
                price=float(price),
                commission=float(commission),
                fees=float(fees),
                realized_pnl_after=float(pos.realized_pnl),
                cash_after=float(self.cash),
            )
        )

    def mark_to_market(self, snapshot: MarketSnapshot, price_source: Literal["mid", "ltp"] = "mid") -> Tuple[float, pd.DataFrame, Dict[str, float]]:
        """
        Compute unrealized PnL and a positions snapshot table.

        Returns: (unrealized_pnl, positions_df, greeks_agg)
        """
        chain = snapshot.options_chain
        unreal = 0.0
        mv_sum = 0.0
        rows = []
        greeks = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

        if chain is None:
            chain = pd.DataFrame()

        for cid, pos in list(self.positions.items()):
            if pos.qty == 0:
                continue
            df = chain[chain["symbol"] == pos.symbol] if not chain.empty else pd.DataFrame()
            mtm = None
            bid = ask = last = mid = None
            if not df.empty:
                r = df.iloc[0]
                bid = r.get("bid")
                ask = r.get("ask")
                last = r.get("last")
                try:
                    bid = float(bid) if bid is not None else None
                except Exception:
                    bid = None
                try:
                    ask = float(ask) if ask is not None else None
                except Exception:
                    ask = None
                try:
                    last = float(last) if last is not None else None
                except Exception:
                    last = None
                if bid is not None and ask is not None and np.isfinite(bid) and np.isfinite(ask):
                    mid = (bid + ask) / 2.0
            if price_source == "mid":
                mtm = mid if mid is not None and np.isfinite(mid) else last
            else:
                mtm = last if last is not None and np.isfinite(last) else mid
            if mtm is None or not np.isfinite(mtm):
                mtm = pos.avg_price

            # For long qty>0: (mtm-avg)*qty*mult
            # For short qty<0: (avg-mtm)*abs(qty)*mult
            if pos.qty > 0:
                u = (float(mtm) - pos.avg_price) * pos.qty * pos.multiplier
                mv = float(mtm) * pos.qty * pos.multiplier
            else:
                u = (pos.avg_price - float(mtm)) * abs(pos.qty) * pos.multiplier
                mv = float(mtm) * pos.qty * pos.multiplier
            unreal += u
            mv_sum += mv

            # Greeks aggregation if present
            if not df.empty:
                for k in ["delta", "gamma", "theta", "vega"]:
                    try:
                        g = float(df.iloc[0].get(k)) if df.iloc[0].get(k) is not None else 0.0
                    except Exception:
                        g = 0.0
                    greeks[k] += g * pos.qty

            rows.append(
                {
                    "timestamp": snapshot.timestamp,
                    "contract_id": pos.contract_id,
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "avg_price": pos.avg_price,
                    "mtm_price": float(mtm),
                    "unrealized_pnl": float(u),
                    "realized_pnl": float(pos.realized_pnl),
                    "hold_bars": pos.hold_bars,
                }
            )

        # attach mv_sum as a synthetic greek field for convenience (runner reads it)
        greeks["_market_value"] = float(mv_sum)
        return float(unreal), pd.DataFrame(rows), greeks

    def step_hold_counters(self) -> None:
        for pos in self.positions.values():
            if pos.qty != 0:
                pos.hold_bars += 1

    def close_all(self, ts: datetime, chain: Optional[pd.DataFrame], exec_fill_fn) -> None:
        """Close all open positions using provided execution fill function (symbol, side, qty)->Fill|None."""
        for cid, pos in list(self.positions.items()):
            if pos.qty == 0:
                continue
            side: Side = "SELL" if pos.qty > 0 else "BUY"
            qty = abs(pos.qty)
            fill = exec_fill_fn(chain, pos.symbol, side, qty)
            if fill is None:
                continue
            self.apply_fill(ts, pos.symbol, side, qty, fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)


def expiry_timestamp_utc(expiry: date, session_end_ist: str = "15:30") -> datetime:
    """Convert an expiry date to a UTC timestamp at IST session close."""
    hour, minute = map(int, session_end_ist.split(":"))
    ist = timezone(timedelta(hours=5, minutes=30))
    dt_ist = datetime(expiry.year, expiry.month, expiry.day, hour, minute, tzinfo=ist)
    return dt_ist.astimezone(timezone.utc)


def intrinsic_value(cp: str, strike: int, spot: float) -> float:
    if cp == "C":
        return max(0.0, spot - float(strike))
    return max(0.0, float(strike) - spot)


