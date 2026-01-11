import pandas as pd
from datetime import datetime, timezone, date

from index_options_bt.portfolio.portfolio import Portfolio
from index_options_bt.data.models import MarketSnapshot


def test_realized_pnl_long_roundtrip():
    p = Portfolio(initial_cash=1000.0, cash=1000.0)
    ts1 = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    ts2 = datetime(2025, 10, 1, 9, 15, 30, tzinfo=timezone.utc)

    # buy 1 at 10
    p.apply_fill(ts1, "NIFTY25NOV2526000CE", "BUY", 1, 10.0, 0.0, 0.0, multiplier=1)
    assert p.cash == 990.0

    # sell 1 at 12 -> +2
    p.apply_fill(ts2, "NIFTY25NOV2526000CE", "SELL", 1, 12.0, 0.0, 0.0, multiplier=1)
    assert p.cash == 1002.0
    assert len(p.closed_trades) == 1
    assert abs(p.closed_trades[0]["realized_pnl"] - 2.0) < 1e-9


def test_unrealized_pnl_mtm():
    p = Portfolio(initial_cash=1000.0, cash=1000.0)
    ts = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    p.apply_fill(ts, "NIFTY25NOV2526000CE", "BUY", 1, 10.0, 0.0, 0.0, multiplier=1)

    chain = pd.DataFrame([{"timestamp": ts, "symbol": "NIFTY25NOV2526000CE", "bid": 11.0, "ask": 13.0, "last": 12.0}])
    snap = MarketSnapshot(timestamp=ts, spot_bar=pd.DataFrame([{"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0, "oi": None}]), options_chain=chain)
    unreal, pos_df, greeks = p.mark_to_market(snap, price_source="mid")
    # mid=12, avg=10 -> +2
    assert abs(unreal - 2.0) < 1e-9
    assert not pos_df.empty


