"""
Standalone script to export backtest results to Excel with IST datetime formatting.

Usage:
    python export_excel.py <run_id>
    python export_excel.py run-20260111-104522-948
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from index_options_bt.run.excel_export import export_run_to_excel


def main():
    """Export a backtest run to Excel"""
    if len(sys.argv) < 2:
        print("Usage: python export_excel.py <run_id>")
        print("\nExample:")
        print("  python export_excel.py run-20260111-104522-948")
        print("\nAvailable runs:")
        
        runs_dir = Path("runs")
        if runs_dir.exists():
            runs = sorted([d.name for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run-")])
            for run in runs[-10:]:  # Show last 10 runs
                print(f"  {run}")
        
        return 1
    
    run_id = sys.argv[1]
    
    try:
        output_path = export_run_to_excel(run_id)
        print(f"\n✓ Excel export successful!")
        print(f"  Output: {output_path}")
        print(f"  Location: reports/{run_id}.xlsx")
        print(f"\nThe Excel file contains:")
        print(f"  - Summary: Run metadata and performance metrics")
        print(f"  - Trades: Complete trade log with IST timestamps")
        print(f"  - Equity Curve: Portfolio value over time")
        print(f"  - Positions: Position snapshots (if available)")
        print(f"\nAll timestamps are in IST (Asia/Kolkata timezone)")
        print(f"\nNote: All Excel reports are saved in the 'reports/' folder")
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print(f"\nMake sure the run ID is correct and the run directory exists.")
        return 1
    except Exception as e:
        print(f"\n✗ Error exporting to Excel: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
