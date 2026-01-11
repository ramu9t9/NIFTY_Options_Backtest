from index_options_bt.risk.manager import RiskManager
from index_options_bt.config.schemas import RiskConfig


def test_risk_manager_caps_by_max_contracts():
    rm = RiskManager(RiskConfig(max_contracts=2))
    d = rm.cap_qty_for_trade(
        desired_qty=10,
        price=None,
        multiplier=50,
        cash_available=1_000_000,
        portfolio_abs_notional=0.0,
        stop_loss_pct=None,
        realized_pnl_today=0.0,
    )
    assert d.approved_qty == 2


def test_risk_manager_rejects_when_limits_need_price_but_missing():
    rm = RiskManager(RiskConfig(max_notional=10_000))
    d = rm.cap_qty_for_trade(
        desired_qty=1,
        price=None,
        multiplier=50,
        cash_available=1_000_000,
        portfolio_abs_notional=0.0,
        stop_loss_pct=None,
        realized_pnl_today=0.0,
    )
    assert d.approved_qty == 0
    assert d.reason == "missing_price_for_limits"


def test_risk_manager_caps_by_max_notional():
    rm = RiskManager(RiskConfig(max_notional=10_000, max_contracts=100))
    # price=100, multiplier=50 => per contract notional ~ 5,000
    d = rm.cap_qty_for_trade(
        desired_qty=10,
        price=100.0,
        multiplier=50,
        cash_available=1_000_000,
        portfolio_abs_notional=0.0,
        stop_loss_pct=None,
        realized_pnl_today=0.0,
    )
    assert d.approved_qty == 2


def test_risk_manager_caps_by_max_loss_per_trade_using_stop_loss_pct():
    rm = RiskManager(RiskConfig(max_loss_per_trade=500, max_contracts=100))
    # price=100, mult=50 => premium=5000/contract
    # stop_loss_pct=5% => risk=250/contract => max_loss 500 => 2 contracts
    d = rm.cap_qty_for_trade(
        desired_qty=10,
        price=100.0,
        multiplier=50,
        cash_available=1_000_000,
        portfolio_abs_notional=0.0,
        stop_loss_pct=5.0,
        realized_pnl_today=0.0,
    )
    assert d.approved_qty == 2


def test_risk_manager_blocks_on_max_loss_per_day():
    rm = RiskManager(RiskConfig(max_loss_per_day=1000, max_contracts=10))
    d = rm.cap_qty_for_trade(
        desired_qty=1,
        price=100.0,
        multiplier=50,
        cash_available=1_000_000,
        portfolio_abs_notional=0.0,
        stop_loss_pct=5.0,
        realized_pnl_today=-1000.0,
    )
    assert d.approved_qty == 0
    assert d.reason == "max_loss_per_day"


