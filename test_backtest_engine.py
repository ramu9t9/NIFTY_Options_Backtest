"""
Test script for backtest engine.
Quick validation before dashboard integration.
"""

import sys
sys.path.append('g:/Projects/NIFTY_Options_Backtest')

from backtest_engine import BacktestEngine
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def test_backtest_engine():
    """Test backtest engine with full date range"""
    
    # Database path
    db_path = r"G:\Projects\Centralize Data Centre\data\nifty_local.db"
    
    # Full date range for validation
    start_date = "2025-08-29"
    end_date = "2026-01-01"
    
    print(f"\n{'='*60}")
    print("Testing Backtest Engine - FULL DATE RANGE")
    print(f"{'='*60}\n")
    print(f"Database: {db_path}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Expected Results:")
    print(f"  - Total Trades: ~114")
    print(f"  - Win Rate: ~57.89%")
    print(f"  - Profit Factor: ~2.24")
    print(f"\n{'='*60}\n")
    
    # Create engine
    config = {
        'candle_interval_seconds': 30,
        'movement_threshold': 0.11,
        'pattern_window_seconds': 60,
        'lot_size': 3750,
        'target_pct': 10.0,
        'stop_pct': 5.0,
        'max_hold_minutes': 3.0
    }
    
    engine = BacktestEngine(db_path, config)
    
    # Run backtest
    print("Starting backtest... This may take a few minutes.\n")
    results = engine.run(start_date, end_date)
    
    # Display results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}\n")
    
    for key, value in results.to_dict().items():
        print(f"{key:.<30} {value}")
    
    # Validation against expected results
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}\n")
    
    expected_trades = 114
    expected_win_rate = 57.89
    expected_profit_factor = 2.24
    
    trades_diff = abs(results.total_trades - expected_trades)
    win_rate_diff = abs(results.win_rate - expected_win_rate)
    pf_diff = abs(results.profit_factor - expected_profit_factor)
    
    print(f"Total Trades: {results.total_trades} (Expected: {expected_trades}, Diff: {trades_diff})")
    print(f"Win Rate: {results.win_rate:.2f}% (Expected: {expected_win_rate}%, Diff: {win_rate_diff:.2f}%)")
    print(f"Profit Factor: {results.profit_factor:.2f} (Expected: {expected_profit_factor}, Diff: {pf_diff:.2f})")
    
    # Check if within tolerance
    trades_ok = trades_diff <= 5
    win_rate_ok = win_rate_diff <= 2.0
    pf_ok = pf_diff <= 0.2
    
    print(f"\nValidation Status:")
    print(f"  Trades: {'✓ PASS' if trades_ok else '✗ FAIL'}")
    print(f"  Win Rate: {'✓ PASS' if win_rate_ok else '✗ FAIL'}")
    print(f"  Profit Factor: {'✓ PASS' if pf_ok else '✗ FAIL'}")
    
    if trades_ok and win_rate_ok and pf_ok:
        print(f"\n{'='*60}")
        print("✓ BACKTEST VALIDATION SUCCESSFUL!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("⚠ BACKTEST VALIDATION NEEDS REVIEW")
        print(f"{'='*60}\n")
    
    # Show sample trades
    print(f"\n{'='*60}")
    print(f"Sample Trades (First 10 of {results.total_trades})")
    print(f"{'='*60}\n")
    
    for trade in results.trades[:10]:
        print(f"  #{trade.order_id}: {trade.option_type} @ {trade.entry_time.strftime('%Y-%m-%d %H:%M')} "
              f"Entry={trade.entry_price:.2f} Exit={trade.exit_price:.2f} "
              f"P&L=₹{trade.net_pnl:,.2f} ({trade.exit_reason})")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    test_backtest_engine()
