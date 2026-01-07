"""
Performance Tracker (CSV)

Reads the paper trades CSV produced by `paper_trading_engine.py` and prints rolling stats:
- trades count, wins/losses, win rate
- total net P&L
- profit factor (sum wins / abs(sum losses))
- avg net per trade

This is intentionally CLI-only (lightweight, no UI dependency).
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

IST = timezone(timedelta(hours=5, minutes=30))


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net": 0.0, "pf": 0.0, "avg": 0.0}
    d = df.copy()
    if "net_pnl" in d.columns:
        d["net_pnl"] = pd.to_numeric(d["net_pnl"], errors="coerce").fillna(0.0)
    else:
        d["net_pnl"] = 0.0
    wins = d[d["net_pnl"] > 0]["net_pnl"].sum()
    losses = d[d["net_pnl"] < 0]["net_pnl"].sum()
    trades = int(len(d))
    win_count = int((d["net_pnl"] > 0).sum())
    loss_count = int((d["net_pnl"] < 0).sum())
    win_rate = (win_count / trades * 100.0) if trades else 0.0
    pf = float(wins / abs(losses)) if losses < 0 else (float("inf") if wins > 0 else 0.0)
    net = float(d["net_pnl"].sum())
    avg = float(net / trades) if trades else 0.0
    return {"trades": trades, "wins": win_count, "losses": loss_count, "win_rate": win_rate, "net": net, "pf": pf, "avg": avg}


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper trading performance tracker (CSV)")
    ap.add_argument("--csv", required=True, help="Path to paper trades CSV")
    ap.add_argument("--refresh-seconds", type=float, default=5.0, help="Refresh interval")
    ap.add_argument("--once", action="store_true", help="Print once and exit")
    args = ap.parse_args()

    expected_win = 58.0
    expected_pf = 2.24

    while True:
        df = _read_csv(args.csv)
        s = _stats(df)
        now_ist = datetime.now(timezone.utc).astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")

        pf_str = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
        print("=" * 72)
        print(f"[{now_ist} IST] {os.path.basename(args.csv)}")
        print(f"Trades: {s['trades']} | Wins: {s['wins']} | Losses: {s['losses']} | WinRate: {s['win_rate']:.2f}% (exp~{expected_win:.2f}%)")
        print(f"Net P&L: {s['net']:.2f} | Avg/Trade: {s['avg']:.2f} | ProfitFactor: {pf_str} (exp~{expected_pf:.2f})")

        if args.once:
            break
        time.sleep(float(args.refresh_seconds))


if __name__ == "__main__":
    main()


