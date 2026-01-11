"""
Backtest runner: orchestrates the backtest execution.

This is the single source of truth for running backtests.
Both CLI and NiceGUI call this function.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from ..config import RunConfig
from ..data import SQLiteDataProvider
from ..strategy import get_strategy, discover_strategies
from ..execution import ChainCache, build_selector, SimpleBidAskExecution
from ..portfolio.portfolio import Portfolio, expiry_timestamp_utc, intrinsic_value
from ..data.symbols import parse_symbol
from ..risk import RiskManager
from .artifacts import RunArtifacts, generate_run_id

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a backtest run"""
    run_id: str
    run_dir: Path
    metrics: Dict[str, Any] = field(default_factory=dict)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: Dict[str, Any] = field(default_factory=dict)


def run_backtest(config: RunConfig, run_id_mode: str = "timestamp") -> RunResult:
    """
    Run a backtest with the given configuration.
    
    This is the single source of truth for running backtests.
    Both CLI and NiceGUI call this function.
    
    Args:
        config: RunConfig instance
        run_id_mode: "deterministic" or "timestamp" for run ID generation
        
    Returns:
        RunResult with metrics, DataFrames, and run_dir
        
    Implements a minimal but real event-driven loop:
    snapshot -> intents -> risk sizing -> selection -> fills -> portfolio -> MTM -> artifacts
    """
    logger.info(f"Starting backtest run...")
    
    # Convert config to dict for artifacts
    config_dict = config.model_dump()
    
    # Generate run ID
    run_id = generate_run_id(config_dict, mode=run_id_mode)
    run_dir_root = Path(config.reporting.run_dir_root)
    artifacts = RunArtifacts(run_dir_root, run_id, config_dict)
    
    try:
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Run directory: {artifacts.run_dir}")
        
        # Write resolved config
        artifacts.write_config_resolved(format="json")
        
        # Discover strategies (if not already done)
        discover_strategies()
        
        # Initialize data provider
        if config.data.provider == "sqlite":
            if not config.data.sqlite_path:
                raise ValueError("sqlite_path is required for SQLite provider")
            
            provider = SQLiteDataProvider(
                sqlite_path=config.data.sqlite_path,
                table=config.data.table,
                symbol_index=config.data.symbol_index,
                bar_size=config.engine.bar_size,
                tz_display=config.engine.tz_display,
                session_start_ist=config.engine.session_start_ist,
                session_end_ist=config.engine.session_end_ist,
            )
        else:
            raise ValueError(f"Unsupported data provider: {config.data.provider}")
        
        # Initialize strategy
        strategy = get_strategy(config.strategy.name, config.strategy.params)
        logger.info(f"Strategy: {config.strategy.name}")
        
        # Get date range
        start_dt = config.engine.get_start_datetime()
        end_dt = config.engine.get_end_datetime()
        logger.info(f"Date range: {start_dt} to {end_dt}")
        
        # Build selector + execution + portfolio
        chain_cache = ChainCache()
        selector = build_selector(
            config.selector,
            chain_cache=chain_cache,
            contract_multiplier=int(getattr(config.selector, "contract_multiplier", 1)),
        )
        exec_model = SimpleBidAskExecution(config.execution)
        portfolio = Portfolio(initial_cash=100_000.0, cash=100_000.0)

        # Logging tables
        selection_rows = []
        positions_rows = []
        equity_rows = []

        # Minimal risk sizing: cap qty per position
        max_contracts = int(config.risk.max_contracts)
        risk_mgr = RiskManager(config.risk)

        # Track realized PnL baseline by UTC day for max_loss_per_day
        realized_pnl_baseline_by_day: Dict[str, float] = {}

        peak_equity = portfolio.cash

        for ts in provider.iter_timestamps(start_dt, end_dt):
            snapshot = provider.get_snapshot(ts)
            if not snapshot.has_spot():
                continue
            spot = snapshot.get_spot_price()
            if spot is None:
                continue

            # Normalized chain for selection/debug fields
            norm_chain = chain_cache.get_or_build(snapshot.options_chain, snapshot.timestamp)

            # --- expiry settlement (before new intents) ---
            for cid, pos in list(portfolio.positions.items()):
                if pos.qty == 0:
                    continue
                try:
                    parsed = parse_symbol(pos.symbol)
                except Exception:
                    continue
                if parsed.kind != "OPT" or parsed.expiry is None or parsed.strike is None or parsed.cp is None:
                    continue
                exp_date = parsed.expiry.date()
                exp_ts = expiry_timestamp_utc(exp_date, session_end_ist=config.engine.session_end_ist or "15:30")
                if snapshot.timestamp >= exp_ts:
                    # cash-settle intrinsic
                    intrinsic = intrinsic_value(parsed.cp, int(parsed.strike), float(spot))
                    # close at intrinsic price via synthetic fill
                    side = "SELL" if pos.qty > 0 else "BUY"
                    qty = abs(pos.qty)
                    portfolio.apply_fill(
                        snapshot.timestamp,
                        pos.symbol,
                        side,  # type: ignore[arg-type]
                        qty,
                        intrinsic,
                        0.0,
                        0.0,
                        multiplier=pos.multiplier,
                    )

            # --- MTM + risk exits (stop/time) ---
            unreal, pos_df, greeks = portfolio.mark_to_market(snapshot, price_source=config.execution.fallback_price)
            mv = float(greeks.get("_market_value", 0.0))
            equity = float(portfolio.cash + mv)
            peak_equity = max(peak_equity, equity)
            dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
            cum_ret = (equity - portfolio.initial_cash) / portfolio.initial_cash if portfolio.initial_cash > 0 else 0.0

            equity_rows.append(
                {
                    "timestamp": snapshot.timestamp,
                    "equity": equity,
                    "cash": float(portfolio.cash),
                    "market_value": mv,
                    "unrealized_pnl": float(unreal),
                    "realized_pnl": float(sum(p.realized_pnl for p in portfolio.positions.values())),
                    "drawdown": float(dd),
                    "cumulative_return": float(cum_ret),
                    "delta": float(greeks.get("delta", 0.0)),
                    "gamma": float(greeks.get("gamma", 0.0)),
                    "theta": float(greeks.get("theta", 0.0)),
                    "vega": float(greeks.get("vega", 0.0)),
                }
            )

            if pos_df is not None and not pos_df.empty:
                positions_rows.extend(pos_df.to_dict("records"))

            # stop-loss / max-hold exits per open position
            for cid, pos in list(portfolio.positions.items()):
                if pos.qty == 0:
                    continue
                # compute mtm for this symbol
                df = snapshot.options_chain
                row = df[df["symbol"] == pos.symbol].iloc[0] if df is not None and not df.empty and (df["symbol"] == pos.symbol).any() else None
                mtm = None
                if row is not None:
                    bid = row.get("bid")
                    ask = row.get("ask")
                    last = row.get("last")
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
                    # Calculate mid - but if bid/ask are 0, mid is 0, not useful
                    # Database has bid=0, ask=0, so we prioritize LTP (last)
                    if bid is not None and ask is not None and bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                    else:
                        mid = None
                    
                    # MTM: prioritize LTP (last) since database has bid=0, ask=0
                    # This matches old engine behavior which uses LTP directly
                    if config.execution.fallback_price == "ltp":
                        mtm = last if last is not None and last > 0 else mid
                    else:  # fallback_price == "mid"
                        mtm = mid if mid is not None and mid > 0 else last
                    
                    # Final fallback
                    if mtm is None or mtm <= 0:
                        mtm = last if last is not None and last > 0 else (mid if mid is not None and mid > 0 else pos.avg_price)
                if mtm is None:
                    mtm = pos.avg_price
                # take-profit
                # NOTE: strategy/config typically expresses TP/SL as percentages (e.g., 10.0 means 10%),
                # but internal math expects a fraction (0.10). Accept both forms for robustness.
                if pos.take_profit_pct is not None and pos.entry_price is not None:
                    tp = float(pos.take_profit_pct)
                    if tp > 1.0:
                        tp = tp / 100.0
                    if pos.qty > 0 and mtm >= pos.entry_price * (1.0 + tp):
                        fill = exec_model.fill(snapshot.options_chain, pos.symbol, "SELL", abs(pos.qty))
                        if fill:
                            portfolio.apply_fill(snapshot.timestamp, pos.symbol, "SELL", abs(pos.qty), fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)
                        continue
                    elif pos.qty < 0 and mtm <= pos.entry_price * (1.0 - tp):
                        fill = exec_model.fill(snapshot.options_chain, pos.symbol, "BUY", abs(pos.qty))
                        if fill:
                            portfolio.apply_fill(snapshot.timestamp, pos.symbol, "BUY", abs(pos.qty), fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)
                        continue
                # stop-loss
                if pos.stop_loss_pct is not None and pos.entry_price is not None:
                    sl = float(pos.stop_loss_pct)
                    if sl > 1.0:
                        sl = sl / 100.0
                    if pos.qty > 0 and mtm <= pos.entry_price * (1.0 - sl):
                        fill = exec_model.fill(snapshot.options_chain, pos.symbol, "SELL", abs(pos.qty))
                        if fill:
                            portfolio.apply_fill(snapshot.timestamp, pos.symbol, "SELL", abs(pos.qty), fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)
                    elif pos.qty < 0 and mtm >= pos.entry_price * (1.0 + sl):
                        fill = exec_model.fill(snapshot.options_chain, pos.symbol, "BUY", abs(pos.qty))
                        if fill:
                            portfolio.apply_fill(snapshot.timestamp, pos.symbol, "BUY", abs(pos.qty), fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)
                # max hold bars
                if pos.max_hold_bars is not None and pos.hold_bars >= int(pos.max_hold_bars):
                    side = "SELL" if pos.qty > 0 else "BUY"
                    fill = exec_model.fill(snapshot.options_chain, pos.symbol, side, abs(pos.qty))
                    if fill:
                        portfolio.apply_fill(snapshot.timestamp, pos.symbol, side, abs(pos.qty), fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)

            # --- Strategy intents ---
            intents = strategy.generate_intents(snapshot, context={"equity": equity})
            for intent in intents:
                if intent.direction == "FLAT":
                    # close all
                    portfolio.close_all(snapshot.timestamp, snapshot.options_chain, exec_model.fill)
                    continue

                # risk sizing
                qty = int(max(1, min(max_contracts, intent.size)))
                intent.size = qty  # mutate for downstream

                legs = selector.select(snapshot, intent, config.selector)
                for leg in legs:
                    # Realized PnL today (UTC date) for max_loss_per_day checks
                    day_key = snapshot.timestamp.date().isoformat()
                    if day_key not in realized_pnl_baseline_by_day:
                        realized_pnl_baseline_by_day[day_key] = float(portfolio.realized_pnl)
                    realized_today = float(portfolio.realized_pnl) - float(realized_pnl_baseline_by_day[day_key])

                    # selection log (best-effort enrich from chain row)
                    row = None
                    if norm_chain is not None and not norm_chain.empty:
                        df = norm_chain[norm_chain["symbol"] == leg.contract.symbol]
                        if not df.empty:
                            row = df.iloc[0]

                    fill = exec_model.fill(snapshot.options_chain, leg.contract.symbol, leg.side, leg.qty)
                    if fill is None:
                        continue

                    # Apply risk limits (may shrink qty or block)
                    stop_loss_pct = intent.metadata.get("stop_loss_pct")
                    take_profit_pct = intent.metadata.get("take_profit_pct")
                    max_hold_bars = intent.metadata.get("max_hold_bars")

                    decision = risk_mgr.cap_qty_for_trade(
                        desired_qty=int(leg.qty),
                        price=float(fill.price),
                        multiplier=int(leg.contract.multiplier),
                        cash_available=float(portfolio.cash),
                        portfolio_abs_notional=float(abs(greeks.get("_market_value", 0.0))),
                        stop_loss_pct=stop_loss_pct,
                        realized_pnl_today=realized_today,
                    )
                    final_qty = int(decision.approved_qty)

                    # Log selection using the FINAL qty (post-risk). If final_qty=0, log and skip.
                    selection_rows.append(
                        {
                            "timestamp": snapshot.timestamp,
                            "strategy": config.strategy.name,
                            "intent_direction": intent.direction,
                            "option_type": intent.option_type,
                            "selected_symbol": leg.contract.symbol,
                            "side": leg.side,
                            "qty": final_qty,
                            "selector_mode": config.selector.mode,
                            "spot": float(spot),
                            "expiry": str(leg.contract.expiry),
                            "strike": leg.contract.strike,
                            "delta": float(row.get("delta")) if row is not None and row.get("delta") is not None else None,
                            "bid": float(row.get("bid")) if row is not None and row.get("bid") is not None else None,
                            "ask": float(row.get("ask")) if row is not None and row.get("ask") is not None else None,
                            "mid": float(row.get("mid")) if row is not None and row.get("mid") is not None else None,
                            "spread_pct": float(row.get("spread_pct")) if row is not None and row.get("spread_pct") is not None else None,
                            "dte": float(row.get("dte")) if row is not None and row.get("dte") is not None else None,
                        }
                    )

                    if final_qty <= 0:
                        continue

                    # pass stop/target/max_hold from strategy params if present
                    # NOTE: if qty was reduced by risk manager, adjust costs accordingly
                    commission = float(config.execution.commission_per_contract) * final_qty
                    fees = float(config.execution.fees_per_contract) * final_qty
                    portfolio.apply_fill(
                        snapshot.timestamp,
                        leg.contract.symbol,
                        leg.side,
                        final_qty,
                        float(fill.price),
                        commission,
                        fees,
                        multiplier=leg.contract.multiplier,
                        stop_loss_pct=stop_loss_pct,
                        take_profit_pct=take_profit_pct,
                        max_hold_bars=max_hold_bars,
                    )

            portfolio.step_hold_counters()

        # Force-close any open positions at end-of-run so trades.csv reflects completed trades.
        # (Otherwise equity moves but trades.csv stays empty because we only export closed round-trips.)
        try:
            if "snapshot" in locals() and snapshot is not None:
                portfolio.close_all(snapshot.timestamp, snapshot.options_chain, exec_model.fill)
        except Exception:
            pass

        provider.close()

        # Build outputs
        equity_curve = pd.DataFrame(equity_rows)
        positions = pd.DataFrame(positions_rows)
        selection = pd.DataFrame(selection_rows)
        trades = pd.DataFrame(portfolio.closed_trades)
        if not trades.empty:
            trades.insert(0, "trade_id", range(1, len(trades) + 1))

        # Metrics
        total_return = float(equity_curve["cumulative_return"].iloc[-1] * 100.0) if not equity_curve.empty else 0.0
        max_drawdown = float(equity_curve["drawdown"].min() * 100.0) if not equity_curve.empty else 0.0
        trade_count = int(len(trades)) if trades is not None and not trades.empty else 0
        win_rate = float((trades["realized_pnl"] > 0).mean() * 100.0) if trade_count > 0 else 0.0
        avg_trade_pnl = float(trades["realized_pnl"].mean()) if trade_count > 0 else 0.0
        sharpe = 0.0
        if not equity_curve.empty and len(equity_curve) > 2:
            rets = equity_curve["equity"].pct_change().dropna()
            if rets.std() != 0:
                sharpe = float((rets.mean() / rets.std()) * np.sqrt(252 * (6.25 * 60 * 60 / pd.to_timedelta(config.engine.bar_size).total_seconds())))

        metrics = {
            "total_trades": trade_count,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe": sharpe,
            "win_rate_pct": win_rate,
            "avg_trade_pnl": avg_trade_pnl,
            "final_equity": float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else float(portfolio.cash),
        }
        
        # Write artifacts
        artifacts.write_manifest({
            "total_trades": trade_count,
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        })
        artifacts.write_equity_curve(equity_curve)
        artifacts.write_trades(trades)
        artifacts.write_positions(positions)
        artifacts.write_selection(selection)
        artifacts.write_metrics(metrics)
        
        logger.info(f"Backtest complete. Run ID: {run_id}")
        
        return RunResult(
            run_id=run_id,
            run_dir=artifacts.run_dir,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades,
            positions=positions,
            config=config_dict,
        )
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise
    finally:
        artifacts.close()

