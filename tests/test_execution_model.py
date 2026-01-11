import pandas as pd
from datetime import datetime, timezone

from index_options_bt.execution.model import SimpleBidAskExecution
from index_options_bt.config.schemas import ExecutionConfig


def test_bidask_fill_with_slippage():
    ts = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    chain = pd.DataFrame(
        [
            {"timestamp": ts, "symbol": "OPT1", "bid": 9.0, "ask": 11.0, "last": 10.0},
        ]
    )
    cfg = ExecutionConfig(fill_on="bidask", fallback_price="mid", slippage_bps=10, commission_per_contract=0.0, fees_per_contract=0.0)
    ex = SimpleBidAskExecution(cfg)
    buy = ex.fill(chain, "OPT1", "BUY", 1)
    sell = ex.fill(chain, "OPT1", "SELL", 1)
    assert buy is not None and sell is not None
    assert abs(buy.price - 11.0 * 1.001) < 1e-9
    assert abs(sell.price - 9.0 * 0.999) < 1e-9


def test_fallback_to_mid_or_ltp():
    ts = datetime(2025, 10, 1, 9, 15, 15, tzinfo=timezone.utc)
    chain = pd.DataFrame([{"timestamp": ts, "symbol": "OPT2", "bid": None, "ask": None, "last": 7.0}])
    cfg = ExecutionConfig(fill_on="bidask", fallback_price="ltp", slippage_bps=0, commission_per_contract=0.0, fees_per_contract=0.0)
    ex = SimpleBidAskExecution(cfg)
    f = ex.fill(chain, "OPT2", "BUY", 1)
    assert f is not None
    assert f.price == 7.0


