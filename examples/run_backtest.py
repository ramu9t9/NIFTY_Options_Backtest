"""
Example runner script demonstrating programmatic backtest execution.

This calls the same run_backtest() function used by CLI and NiceGUI.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index_options_bt.config import load_config, RunConfig
from index_options_bt.run.runner import run_backtest


def main():
    """Run example backtest"""
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "breakout_nifty_sqlite.yaml"
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please create a config file first or update the path.")
        return 1
    
    try:
        # Load configuration
        config = load_config(str(config_path))
        
        # Optionally override programmatically
        # config.engine.start = "2025-10-01"
        # config.engine.end = "2025-10-15"
        # config.execution.slippage_bps = 3.0
        
        # Run backtest
        print(f"Running backtest with config: {config_path}")
        result = run_backtest(config, run_id_mode="timestamp")
        
        # Print summary
        print("\n" + "=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)
        print(f"Run ID: {result.run_id}")
        print(f"Run Directory: {result.run_dir}")
        print("-" * 70)
        metrics = result.metrics
        print(f"Total Return: {metrics.get('total_return_pct', 0.0):.2f}%")
        print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0.0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe', 0.0):.2f}")
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate_pct', 0.0):.2f}%")
        print(f"Avg Trade P&L: INR {metrics.get('avg_trade_pnl', 0.0):,.2f}")
        print("=" * 70)
        
        print(f"\nResults saved to: {result.run_dir}")
        print("Files:")
        print(f"  - config_resolved.json")
        print(f"  - manifest.json")
        print(f"  - equity_curve.csv")
        print(f"  - trades.csv")
        print(f"  - positions.csv")
        print(f"  - metrics.json")
        print(f"  - report.png (if matplotlib available)")
        print(f"  - run.log")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

