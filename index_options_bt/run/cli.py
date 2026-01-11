"""
CLI entrypoint for running backtests.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from ..config import load_config, apply_env_overrides, apply_cli_overrides, RunConfig
from ..strategy import list_strategies, discover_strategies
from .runner import run_backtest, RunResult

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_summary(result: RunResult):
    """Print backtest summary to console"""
    metrics = result.metrics
    
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    print(f"Run ID: {result.run_id}")
    print(f"Run Directory: {result.run_dir}")
    print("-" * 70)
    print(f"Total Return: {metrics.get('total_return_pct', 0.0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe', 0.0):.2f}")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate_pct', 0.0):.2f}%")
    print(f"Avg Trade P&L: INR {metrics.get('avg_trade_pnl', 0.0):,.2f}")
    print("=" * 70 + "\n")


def cmd_list_strategies():
    """List all available strategies"""
    discover_strategies()
    strategies = list_strategies()
    
    print("\n" + "=" * 70)
    print("AVAILABLE STRATEGIES")
    print("=" * 70)
    if strategies:
        for strategy in strategies:
            print(f"  - {strategy}")
    else:
        print("  (No strategies registered)")
        print("\n  Make sure strategy modules are imported in index_options_bt/strategy/__init__.py")
    print("=" * 70 + "\n")


def cmd_dry_run(config: RunConfig, run_id_mode: str):
    """Dry run: resolve config and print run ID without executing"""
    from .artifacts import generate_run_id
    
    config_dict = config.model_dump()
    run_id = generate_run_id(config_dict, mode=run_id_mode)
    
    print("\n" + "=" * 70)
    print("DRY RUN - Configuration Resolved")
    print("=" * 70)
    print(f"Run ID (mode: {run_id_mode}): {run_id}")
    print(f"Strategy: {config.strategy.name}")
    print(f"Date Range: {config.engine.start} to {config.engine.end}")
    print(f"Bar Size: {config.engine.bar_size}")
    print(f"Data Provider: {config.data.provider}")
    if config.data.provider == "sqlite":
        print(f"SQLite Path: {config.data.sqlite_path}")
    print("=" * 70 + "\n")
    
    return run_id


def cmd_run(
    config_path: Optional[str],
    start: Optional[str],
    end: Optional[str],
    strategy: Optional[str],
    sets: List[str],
    run_id_mode: str,
    profile: bool = False,
) -> RunResult:
    """Run backtest"""
    # Load base config
    if config_path:
        config = load_config(config_path)
    else:
        raise ValueError("--config is required")
    
    # Apply environment overrides
    config = apply_env_overrides(config)
    
    # Apply CLI overrides
    if sets:
        config = apply_cli_overrides(config, sets)
    
    # Override start/end if provided
    if start:
        config.engine.start = start
    if end:
        config.engine.end = end
    
    # Override strategy if provided
    if strategy:
        config.strategy.name = strategy
    
    # Discover strategies
    discover_strategies()
    
    # Profile if requested
    if profile:
        import cProfile
        import pstats
        import io
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = run_backtest(config, run_id_mode=run_id_mode)
        finally:
            profiler.disable()
            
            # Print profile stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions
            print("\n" + "=" * 70)
            print("PROFILING RESULTS (Top 20 functions)")
            print("=" * 70)
            print(s.getvalue())
            print("=" * 70 + "\n")
        
        return result
    else:
        return run_backtest(config, run_id_mode=run_id_mode)


def main():
    """Main CLI entrypoint"""
    parser = argparse.ArgumentParser(
        description="Index Options Backtest Engine - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python -m index_options_bt.run --config configs/breakout_nifty_sqlite.yaml
  
  # Override date range
  python -m index_options_bt.run --config configs/breakout_nifty_sqlite.yaml --start 2025-10-01 --end 2025-10-15
  
  # Override strategy and config values
  python -m index_options_bt.run --config configs/breakout_nifty_sqlite.yaml --strategy breakout --set execution.slippage_bps=5
  
  # Dry run (resolve config without executing)
  python -m index_options_bt.run --config configs/breakout_nifty_sqlite.yaml --dry-run
  
  # List available strategies
  python -m index_options_bt.run --list-strategies
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (YAML or JSON)",
    )
    
    parser.add_argument(
        "--start",
        type=str,
        help="Override start date (ISO format, e.g., 2025-10-01)",
    )
    
    parser.add_argument(
        "--end",
        type=str,
        help="Override end date (ISO format, e.g., 2025-10-15)",
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        help="Override strategy name",
    )
    
    parser.add_argument(
        "--set",
        action="append",
        dest="sets",
        metavar="KEY=VALUE",
        help="Override config value (can be used multiple times). Use nested keys: execution.slippage_bps=2",
    )
    
    parser.add_argument(
        "--run-id-mode",
        choices=["deterministic", "timestamp"],
        default="timestamp",
        help="Run ID generation mode (default: timestamp)",
    )
    
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List all available strategies and exit",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and print run ID without executing",
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling (timing breakdown)",
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # List strategies command
    if args.list_strategies:
        cmd_list_strategies()
        return 0
    
    # Dry run command
    if args.dry_run:
        if not args.config:
            print("ERROR: --config is required for --dry-run", file=sys.stderr)
            return 1
        
        try:
            config = load_config(args.config)
            config = apply_env_overrides(config)
            if args.sets:
                config = apply_cli_overrides(config, args.sets)
            if args.start:
                config.engine.start = args.start
            if args.end:
                config.engine.end = args.end
            if args.strategy:
                config.strategy.name = args.strategy
            
            cmd_dry_run(config, args.run_id_mode)
            return 0
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            logger.exception("Dry run failed")
            return 1
    
    # Run command
    if not args.config:
        print("ERROR: --config is required", file=sys.stderr)
        parser.print_help()
        return 1
    
    try:
        result = cmd_run(
            config_path=args.config,
            start=args.start,
            end=args.end,
            strategy=args.strategy,
            sets=args.sets or [],
            run_id_mode=args.run_id_mode,
            profile=args.profile,
        )
        
        print_summary(result)
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        logger.exception("Backtest failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

