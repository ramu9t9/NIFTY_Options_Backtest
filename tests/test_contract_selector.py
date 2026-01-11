import pandas as pd
from datetime import datetime, timezone, date

from index_options_bt.execution.chain_cache import ChainCache
from index_options_bt.execution.selectors import build_selector
from index_options_bt.strategy.base import Intent
from index_options_bt.data.models import MarketSnapshot
from index_options_bt.config.schemas import SelectorConfig, StrikeWindow, LiquidityFilters


def _snapshot(ts: datetime, spot: float, chain: pd.DataFrame) -> MarketSnapshot:
    spot_bar = pd.DataFrame([{"open": spot, "high": spot, "low": spot, "close": spot, "volume": 0, "oi": None}])
    return MarketSnapshot(timestamp=ts, spot_bar=spot_bar, options_chain=chain)


def test_atm_selector_picks_nearest_strike():
    ts = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    chain = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "X1", "expiry": date(2025, 10, 2), "strike": 100, "cp": "C", "bid": 9, "ask": 11, "last": 10, "delta": 0.5, "oi": 1000, "volume": 100},
            {"timestamp": ts, "symbol": "X2", "expiry": date(2025, 10, 2), "strike": 110, "cp": "C", "bid": 9, "ask": 11, "last": 10, "delta": 0.4, "oi": 1000, "volume": 100},
        ]
    )
    cfg = SelectorConfig(mode="atm", expiry_preference="nearest", strike_window=StrikeWindow(kind="count", value=10), liquidity=LiquidityFilters(min_bid=1, max_spread_pct=1, min_oi=0, min_volume=0))
    selector = build_selector(cfg, ChainCache())
    intent = Intent(timestamp=ts, direction="LONG", option_type="CALL", size=1)
    legs = selector.select(_snapshot(ts, 101, chain), intent, cfg)
    assert len(legs) == 1
    assert legs[0].contract.symbol == "X1"


def test_dte_selector_prefers_target_dte():
    ts = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    chain = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "E1", "expiry": date(2025, 10, 2), "strike": 100, "cp": "P", "bid": 9, "ask": 11, "last": 10, "delta": -0.5, "oi": 1000, "volume": 100},
            {"timestamp": ts, "symbol": "E2", "expiry": date(2025, 10, 10), "strike": 100, "cp": "P", "bid": 9, "ask": 11, "last": 10, "delta": -0.5, "oi": 1000, "volume": 100},
        ]
    )
    cfg = SelectorConfig(mode="dte", target_dte=9, expiry_preference="nearest", strike_window=StrikeWindow(kind="count", value=10), liquidity=LiquidityFilters(min_bid=1, max_spread_pct=1, min_oi=0, min_volume=0))
    selector = build_selector(cfg, ChainCache())
    intent = Intent(timestamp=ts, direction="LONG", option_type="PUT", size=1)
    legs = selector.select(_snapshot(ts, 100, chain), intent, cfg)
    assert len(legs) == 1
    assert legs[0].contract.symbol == "E2"


def test_delta_selector_targets_delta_and_liquidity_filters():
    ts = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    chain = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "D1", "expiry": date(2025, 10, 2), "strike": 100, "cp": "C", "bid": 0.5, "ask": 5.0, "last": 3.0, "delta": 0.25, "oi": 1000, "volume": 100},
            {"timestamp": ts, "symbol": "D2", "expiry": date(2025, 10, 2), "strike": 105, "cp": "C", "bid": 2.0, "ask": 2.2, "last": 2.1, "delta": 0.30, "oi": 1000, "volume": 100},
        ]
    )
    cfg = SelectorConfig(
        mode="delta",
        target_delta=0.25,
        expiry_preference="nearest",
        strike_window=StrikeWindow(kind="count", value=10),
        liquidity=LiquidityFilters(min_bid=1.0, max_spread_pct=0.5, min_oi=0, min_volume=0),
    )
    selector = build_selector(cfg, ChainCache())
    intent = Intent(timestamp=ts, direction="LONG", option_type="CALL", size=1)
    legs = selector.select(_snapshot(ts, 102, chain), intent, cfg)
    # D1 filtered out by min_bid; D2 remains
    assert len(legs) == 1
    assert legs[0].contract.symbol == "D2"


