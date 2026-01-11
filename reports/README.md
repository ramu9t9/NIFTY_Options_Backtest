# Reports Folder

This folder contains Excel exports of backtest results.

## Structure

Each backtest run exports to: `reports/<run_id>.xlsx`

Example:
```
reports/
├── run-20260111-104522-948.xlsx
├── run-20260111-104544-353.xlsx
└── run-20260111-105030-123.xlsx
```

## Excel File Contents

Each Excel file contains multiple sheets:

1. **Summary**: Run metadata and performance metrics
2. **Trades**: Complete trade log with IST timestamps
3. **Equity Curve**: Portfolio value over time
4. **Positions**: Position snapshots (if available)
5. **Contract Selection**: Debug info (optional)

## Timezone

All timestamps in Excel files are in **IST (Asia/Kolkata)** timezone, converted from UTC.

## Usage

Export any backtest run:
```bash
python export_excel.py <run_id>
```

The Excel file will be automatically saved in this `reports/` folder.
