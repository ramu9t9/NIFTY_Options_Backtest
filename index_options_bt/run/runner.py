"""
Backtest runner: orchestrates the backtest execution.

This is the single source of truth for running backtests.
Both CLI and NiceGUI call this function.
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

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
class _PendingOrder:
    signal_ts: datetime
    execute_ts: datetime
    strategy: str
    intent_direction: str
    option_type: Optional[str]
    contract_symbol: str
    side: str
    desired_qty: int
    multiplier: int
    selector_mode: str
    spot_at_signal: float
    expiry: Optional[str]
    strike: Optional[float]
    qty_mode: str = "fixed"  # "fixed" or "all" (all = use current position qty at execution time)
    # Carry through trade management metadata (TP/SL/time stops)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
                volume_mode=getattr(config.data, "volume_mode", "incremental"),
            )
        else:
            raise ValueError(f"Unsupported data provider: {config.data.provider}")
        
        # Initialize strategy
        strategy = get_strategy(config.strategy.name, config.strategy.params)
        logger.info(f"Strategy: {config.strategy.name}")

        # Enforce BUY-only engine defaults (strategy can remain in repo, but cannot be run unless explicitly overridden)
        if bool(getattr(config.engine, "buy_only", True)) and str(config.strategy.name).lower() in {"strangle"}:
            raise ValueError("BUY-only engine: option-selling strategies (e.g., 'strangle') are disabled. Choose a buy-only strategy.")
        
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

        # Pre-compute timestamps so we can (a) show progress and (b) schedule next-bar entries.
        timestamps = list(provider.iter_timestamps(start_dt, end_dt))
        total_bars = len(timestamps)
        t0 = time.time()
        last_progress_log = t0
        pending: List[_PendingOrder] = []

        logger.info(f"Total bars to simulate: {total_bars}")

        # Prefer a real progress bar in CLI when available.
        use_tqdm = bool(os.environ.get("INDEX_BT_TQDM", "1") != "0") and sys.stderr.isatty()
        pbar = None
        if use_tqdm:
            try:
                from tqdm.auto import tqdm  # type: ignore
                pbar = tqdm(total=total_bars, desc="Simulating", unit="bar", dynamic_ncols=True)
            except Exception:
                pbar = None

        for i, ts in enumerate(timestamps):
            next_ts = timestamps[i + 1] if i + 1 < total_bars else None
            bar_seconds = float(pd.to_timedelta(config.engine.bar_size).total_seconds())

            # Backend progress
            if pbar is not None:
                pbar.update(1)
                if i % 200 == 0:
                    pbar.set_postfix(ts=ts.isoformat(timespec="seconds"))
            else:
                # Fallback: log every ~5s (no tqdm installed or non-interactive output)
                now = time.time()
                if now - last_progress_log >= 5.0 and total_bars > 0:
                    pct = (i / total_bars) * 100.0
                    elapsed = now - t0
                    speed = (i / elapsed) if elapsed > 0 else 0.0
                    eta_s = ((total_bars - i) / speed) if speed > 0 else None
                    eta_str = f"{eta_s/60.0:.1f}m" if eta_s is not None else "n/a"
                    logger.info(f"Progress: {pct:.1f}% ({i}/{total_bars}) | speed={speed:.1f} bars/s | ETA={eta_str} | ts={ts.isoformat()}")
                    last_progress_log = now

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

            # --- execute pending entries scheduled for this bar (entry on next bar) ---
            if pending:
                to_run = [p for p in pending if p.execute_ts == snapshot.timestamp]
                if to_run:
                    # Keep remaining pending
                    pending = [p for p in pending if p.execute_ts != snapshot.timestamp]
                    for p in to_run:
                        # Determine qty
                        desired_qty = int(p.desired_qty)
                        if str(p.qty_mode).lower() == "all":
                            # close current position size at execution time (BUY-only => long positions only)
                            cid = option_contract_id(p.contract_symbol)
                            pos = portfolio.positions.get(cid)
                            desired_qty = int(abs(pos.qty)) if pos is not None else 0
                        if desired_qty <= 0:
                            continue

                        # Realistic pricing: next-bar open tick (first tick inside this bar) + slippage
                        bar_start = snapshot.timestamp - pd.Timedelta(seconds=bar_seconds)
                        fill = exec_model.fill_on_next_bar_open(
                            chain=snapshot.options_chain,
                            options_ticks=getattr(snapshot, "options_ticks", None),
                            symbol=p.contract_symbol,
                            side=str(p.side).upper(),  # type: ignore[arg-type]
                            qty=int(desired_qty),
                            bar_start_utc=bar_start.to_pydatetime(),
                            bar_end_utc=snapshot.timestamp,
                        )
                        if fill is None:
                            continue

                        # Realized PnL today (UTC date) for max_loss_per_day checks
                        day_key = snapshot.timestamp.date().isoformat()
                        if day_key not in realized_pnl_baseline_by_day:
                            realized_pnl_baseline_by_day[day_key] = float(portfolio.realized_pnl)
                        realized_today = float(portfolio.realized_pnl) - float(realized_pnl_baseline_by_day[day_key])

                        decision = risk_mgr.cap_qty_for_trade(
                            desired_qty=int(desired_qty),
                            price=float(fill.price),
                            multiplier=int(p.multiplier),
                            cash_available=float(portfolio.cash),
                            portfolio_abs_notional=float(abs(0.0)),
                            stop_loss_pct=p.metadata.get("stop_loss_pct"),
                            realized_pnl_today=realized_today,
                        )
                        final_qty = int(decision.approved_qty)

                        selection_rows.append(
                            {
                                "timestamp": snapshot.timestamp,
                                "signal_timestamp": p.signal_ts,
                                "execution_timestamp": snapshot.timestamp,
                                "strategy": p.strategy,
                                "intent_direction": p.intent_direction,
                                "option_type": p.option_type,
                                "selected_symbol": p.contract_symbol,
                                "side": p.side,
                                "qty": final_qty,
                                "selector_mode": p.selector_mode,
                                "spot": float(spot),
                                "spot_at_signal": float(p.spot_at_signal),
                                "expiry": p.expiry,
                                "strike": p.strike,
                            }
                        )

                        if final_qty <= 0:
                            continue

                        commission = float(config.execution.commission_per_contract) * final_qty
                        fees = float(config.execution.fees_per_contract) * final_qty
                        portfolio.apply_fill(
                            snapshot.timestamp,
                            p.contract_symbol,
                            p.side,
                            final_qty,
                            float(fill.price),
                            commission,
                            fees,
                            multiplier=int(p.multiplier),
                            stop_loss_pct=p.metadata.get("stop_loss_pct"),
                            take_profit_pct=p.metadata.get("take_profit_pct"),
                            take_profit_pct_1=p.metadata.get("take_profit_pct_1"),
                            take_profit_pct_2=p.metadata.get("take_profit_pct_2"),
                            tp1_fraction=p.metadata.get("tp1_fraction"),
                            time_stop_bars=p.metadata.get("time_stop_bars"),
                            min_profit_pct_by_time_stop=p.metadata.get("min_profit_pct_by_time_stop"),
                            max_hold_bars=p.metadata.get("max_hold_bars"),
                        )

            # --- MTM + risk exits (stop/time) ---
            unreal, pos_df, greeks = portfolio.mark_to_market(snapshot, price_source="ltp")
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
                # compute mtm for this symbol (LTP-only; bid/ask are always 0 in DB)
                df = snapshot.options_chain
                row = df[df["symbol"] == pos.symbol].iloc[0] if df is not None and not df.empty and (df["symbol"] == pos.symbol).any() else None
                mtm = None
                if row is not None:
                    last = row.get("last")
                    try:
                        last = float(last) if last is not None else None
                    except Exception:
                        last = None
                    mtm = last if last is not None and last > 0 else pos.avg_price
                if mtm is None:
                    mtm = pos.avg_price

                # Track MAE/MFE in premium % relative to entry (signed by entry side).
                if pos.entry_price is not None and pos.entry_price > 0 and getattr(pos, "entry_side", None) is not None:
                    side = pos.entry_side
                    if side == "BUY":
                        signed_ret = (float(mtm) / float(pos.entry_price) - 1.0) * 100.0
                    else:
                        signed_ret = (float(pos.entry_price) / float(mtm) - 1.0) * 100.0 if float(mtm) > 0 else 0.0
                    # init extrema if missing
                    if getattr(pos, "mae_price", None) is None:
                        pos.mae_price = float(mtm)
                    if getattr(pos, "mfe_price", None) is None:
                        pos.mfe_price = float(mtm)
                    if signed_ret > float(getattr(pos, "mfe_pct", 0.0)):
                        pos.mfe_pct = float(signed_ret)
                        pos.mfe_price = float(mtm)
                    if signed_ret < float(getattr(pos, "mae_pct", 0.0)):
                        pos.mae_pct = float(signed_ret)
                        pos.mae_price = float(mtm)

                # take-profit
                # NOTE: strategy/config typically expresses TP/SL as percentages (e.g., 10.0 means 10%),
                # but internal math expects a fraction (0.10). Accept both forms for robustness.
                # TP1 (optional partial)
                if pos.take_profit_pct_1 is not None and pos.entry_price is not None and not getattr(pos, "tp1_done", False):
                    tp1 = float(pos.take_profit_pct_1)
                    if tp1 > 1.0:
                        tp1 = tp1 / 100.0
                    if pos.qty > 0 and mtm >= pos.entry_price * (1.0 + tp1):
                        # partial close only if qty >= 2, else mark done and let TP2 handle full exit
                        frac = float(pos.tp1_fraction) if pos.tp1_fraction is not None else 0.5
                        frac = min(max(frac, 0.0), 1.0)
                        close_qty = abs(pos.qty)
                        if close_qty >= 2 and frac > 0:
                            q = max(1, int(round(close_qty * frac)))
                            fill = exec_model.fill(snapshot.options_chain, pos.symbol, "SELL", q)
                            if fill:
                                portfolio.apply_fill(snapshot.timestamp, pos.symbol, "SELL", q, fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)
                                # mark TP1 done
                                try:
                                    pos.tp1_done = True
                                except Exception:
                                    pass
                        else:
                            try:
                                pos.tp1_done = True
                            except Exception:
                                pass
                # TP2 / single TP (full exit)
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
                # time stop: if not in profit by bar N, exit
                if getattr(pos, "time_stop_bars", None) is not None and pos.entry_price is not None:
                    ts_bars = int(pos.time_stop_bars)  # type: ignore[arg-type]
                    if ts_bars > 0 and pos.hold_bars >= ts_bars:
                        minp = float(pos.min_profit_pct_by_time_stop) if pos.min_profit_pct_by_time_stop is not None else 0.0
                        if minp > 1.0:
                            minp = minp / 100.0
                        # long only expected; keep generic
                        if pos.qty > 0 and mtm <= pos.entry_price * (1.0 + minp):
                            fill = exec_model.fill(snapshot.options_chain, pos.symbol, "SELL", abs(pos.qty))
                            if fill:
                                portfolio.apply_fill(snapshot.timestamp, pos.symbol, "SELL", abs(pos.qty), fill.price, fill.commission, fill.fees, multiplier=pos.multiplier)
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

                # BUY-only enforcement: disallow SHORT intents (no option selling / short positions)
                if bool(getattr(config.engine, "buy_only", True)) and str(intent.direction).upper() == "SHORT":
                    logger.warning("BUY-only engine: ignoring SHORT intent generated by strategy.")
                    continue

                # risk sizing
                qty = int(max(1, min(max_contracts, intent.size)))
                intent.size = qty  # mutate for downstream

                legs = selector.select(snapshot, intent, config.selector)
                for leg in legs:
                    # BUY-only enforcement: do not allow SELL legs as *entries* (SELL is allowed only for exits).
                    if bool(getattr(config.engine, "buy_only", True)) and str(leg.side).upper() == "SELL":
                        logger.warning("BUY-only engine: ignoring SELL leg from selector (entries must be BUY).")
                        continue
                    # Apply risk limits (may shrink qty or block) - evaluated at execution time for next-bar entry.
                    stop_loss_pct = intent.metadata.get("stop_loss_pct")
                    take_profit_pct = intent.metadata.get("take_profit_pct")
                    max_hold_bars = intent.metadata.get("max_hold_bars")
                    take_profit_pct_1 = intent.metadata.get("take_profit_pct_1")
                    take_profit_pct_2 = intent.metadata.get("take_profit_pct_2")
                    tp1_fraction = intent.metadata.get("tp1_fraction")
                    time_stop_bars = intent.metadata.get("time_stop_bars")
                    min_profit_pct_by_time_stop = intent.metadata.get("min_profit_pct_by_time_stop")

                    # Execute on next bar after signal (default). If we're at the last bar, skip scheduling.
                    if bool(getattr(config.engine, "entry_on_next_bar", True)):
                        if next_ts is None:
                            continue
                        pending.append(
                            _PendingOrder(
                                signal_ts=snapshot.timestamp,
                                execute_ts=next_ts,
                                strategy=config.strategy.name,
                                intent_direction=str(intent.direction),
                                option_type=str(intent.option_type) if intent.option_type is not None else None,
                                contract_symbol=str(leg.contract.symbol),
                                side=str(leg.side),
                                desired_qty=int(leg.qty),
                                multiplier=int(leg.contract.multiplier),
                                selector_mode=str(config.selector.mode),
                                spot_at_signal=float(spot),
                                expiry=str(leg.contract.expiry) if leg.contract.expiry is not None else None,
                                strike=float(leg.contract.strike) if leg.contract.strike is not None else None,
                                metadata={
                                    "stop_loss_pct": stop_loss_pct,
                                    "take_profit_pct": take_profit_pct,
                                    "take_profit_pct_1": take_profit_pct_1,
                                    "take_profit_pct_2": take_profit_pct_2,
                                    "tp1_fraction": tp1_fraction,
                                    "time_stop_bars": time_stop_bars,
                                    "min_profit_pct_by_time_stop": min_profit_pct_by_time_stop,
                                    "max_hold_bars": max_hold_bars,
                                },
                            )
                        )
                    else:
                        # Legacy behavior: execute on same bar
                        fill = exec_model.fill(snapshot.options_chain, leg.contract.symbol, leg.side, leg.qty)
                        if fill is None:
                            continue
                        decision = risk_mgr.cap_qty_for_trade(
                            desired_qty=int(leg.qty),
                            price=float(fill.price),
                            multiplier=int(leg.contract.multiplier),
                            cash_available=float(portfolio.cash),
                            portfolio_abs_notional=float(abs(greeks.get("_market_value", 0.0))),
                            stop_loss_pct=stop_loss_pct,
                            realized_pnl_today=0.0,
                        )
                        final_qty = int(decision.approved_qty)
                        selection_rows.append(
                            {
                                "timestamp": snapshot.timestamp,
                                "signal_timestamp": snapshot.timestamp,
                                "execution_timestamp": snapshot.timestamp,
                                "strategy": config.strategy.name,
                                "intent_direction": intent.direction,
                                "option_type": intent.option_type,
                                "selected_symbol": leg.contract.symbol,
                                "side": leg.side,
                                "qty": final_qty,
                                "selector_mode": config.selector.mode,
                                "spot": float(spot),
                                "spot_at_signal": float(spot),
                                "expiry": str(leg.contract.expiry),
                                "strike": leg.contract.strike,
                            }
                        )
                        if final_qty <= 0:
                            continue
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
                            take_profit_pct_1=take_profit_pct_1,
                            take_profit_pct_2=take_profit_pct_2,
                            tp1_fraction=tp1_fraction,
                            time_stop_bars=time_stop_bars,
                            min_profit_pct_by_time_stop=min_profit_pct_by_time_stop,
                            max_hold_bars=max_hold_bars,
                        )

            portfolio.step_hold_counters()

        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

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

        # Enrich trades with return metrics + hour buckets (for reporting)
        if trades is not None and not trades.empty:
            # signed premium return % (BUY: (exit-entry)/entry, SELL: (entry-exit)/entry)
            side = trades.get("entry_side")
            if side is None:
                trades["entry_side"] = "BUY"
            trades["entry_side"] = trades["entry_side"].fillna("BUY")
            ep = pd.to_numeric(trades["entry_price"], errors="coerce")
            xp = pd.to_numeric(trades["exit_price"], errors="coerce")
            trades["return_pct"] = np.where(
                trades["entry_side"].astype(str).str.upper() == "BUY",
                (xp / ep - 1.0) * 100.0,
                (ep / xp - 1.0) * 100.0,
            )

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

        # Scalp-specific metrics + extra CSV reports
        if trades is not None and not trades.empty:
            # averages
            if "hold_bars" in trades.columns:
                metrics["avg_hold_bars"] = float(pd.to_numeric(trades["hold_bars"], errors="coerce").mean())
            if "mae_pct" in trades.columns:
                metrics["avg_mae_pct"] = float(pd.to_numeric(trades["mae_pct"], errors="coerce").mean())
            if "mfe_pct" in trades.columns:
                metrics["avg_mfe_pct"] = float(pd.to_numeric(trades["mfe_pct"], errors="coerce").mean())
            if "return_pct" in trades.columns:
                metrics["avg_return_pct"] = float(pd.to_numeric(trades["return_pct"], errors="coerce").mean())

            # win rate by hour bucket (in display timezone)
            try:
                tz = ZoneInfo(str(config.engine.tz_display))
            except Exception:
                tz = ZoneInfo("Asia/Kolkata")
            et = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
            trades["_entry_time_local"] = et.dt.tz_convert(tz)
            trades["_entry_hour"] = trades["_entry_time_local"].dt.hour
            by_hour = (
                trades.groupby("_entry_hour", dropna=True)
                .agg(
                    trades=("trade_id", "count"),
                    wins=("realized_pnl", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
                    win_rate_pct=("realized_pnl", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean() * 100.0) if len(x) else 0.0),
                    avg_pnl=("realized_pnl", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else 0.0),
                    avg_return_pct=("return_pct", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else 0.0),
                )
                .reset_index()
                .rename(columns={"_entry_hour": "hour_local"})
                .sort_values("hour_local")
            )
            by_hour.to_csv(artifacts.run_dir / "win_rate_by_hour.csv", index=False)

            # return distribution (simple bins)
            r = pd.to_numeric(trades["return_pct"], errors="coerce")
            bins = [-1e9, -30, -20, -10, -5, 0, 5, 10, 20, 30, 50, 1e9]
            labels = ["<=-30", "-30..-20", "-20..-10", "-10..-5", "-5..0", "0..5", "5..10", "10..20", "20..30", "30..50", ">=50"]
            trades["_ret_bin"] = pd.cut(r, bins=bins, labels=labels, include_lowest=True)
            dist = (
                trades.groupby("_ret_bin", dropna=False)
                .agg(
                    trades=("trade_id", "count"),
                    win_rate_pct=("realized_pnl", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean() * 100.0) if len(x) else 0.0),
                    avg_pnl=("realized_pnl", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else 0.0),
                    avg_return_pct=("return_pct", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else 0.0),
                    avg_mae_pct=("mae_pct", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else 0.0),
                    avg_mfe_pct=("mfe_pct", lambda x: float(pd.to_numeric(x, errors="coerce").mean()) if len(x) else 0.0),
                )
                .reset_index()
                .rename(columns={"_ret_bin": "return_bin"})
            )
            dist.to_csv(artifacts.run_dir / "return_distribution.csv", index=False)
            metrics["return_distribution_bins"] = labels
        
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

