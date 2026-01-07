import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TradeStore:
    """
    Simple SQLite persistence for paper trades.
    - Writes completed trades
    - Supports querying recent/history for dashboard
    """

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id INTEGER,
                    trend_number INTEGER,
                    predicted_direction TEXT,
                    actual_direction TEXT,
                    option_symbol TEXT,
                    option_type TEXT,
                    strike INTEGER,
                    expiry TEXT,
                    patterns_count INTEGER,
                    patterns TEXT,
                    spot_price REAL,
                    rally_start_utc TEXT,
                    signal_time_utc TEXT,
                    planned_entry_time_utc TEXT,
                    entry_time_utc TEXT,
                    exit_time_utc TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    exit_reason TEXT,
                    target_hit INTEGER,
                    pnl_points REAL,
                    gross_pnl REAL,
                    transaction_cost REAL,
                    net_pnl REAL,
                    hold_time_minutes REAL,
                    created_at_utc TEXT
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time_utc);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(option_symbol);")
            self._conn.commit()

    def insert_trade(self, trade_dict: Dict[str, Any]) -> None:
        cols = [
            "order_id",
            "trend_number",
            "predicted_direction",
            "actual_direction",
            "option_symbol",
            "option_type",
            "strike",
            "expiry",
            "patterns_count",
            "patterns",
            "spot_price",
            "rally_start_utc",
            "signal_time_utc",
            "planned_entry_time_utc",
            "entry_time_utc",
            "exit_time_utc",
            "entry_price",
            "exit_price",
            "exit_reason",
            "target_hit",
            "pnl_points",
            "gross_pnl",
            "transaction_cost",
            "net_pnl",
            "hold_time_minutes",
            "created_at_utc",
        ]

        row = {c: trade_dict.get(c) for c in cols}
        if row.get("created_at_utc") is None:
            row["created_at_utc"] = datetime.utcnow().isoformat(timespec="seconds")

        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO trades ({', '.join(cols)}) VALUES ({placeholders})"

        with self._lock:
            self._conn.execute(sql, [row[c] for c in cols])
            self._conn.commit()

    def list_trades(self, limit: int = 500, since_utc: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = int(limit)
        with self._lock:
            cur = self._conn.cursor()
            if since_utc:
                cur.execute(
                    """
                    SELECT * FROM trades
                    WHERE exit_time_utc IS NOT NULL AND exit_time_utc >= ?
                    ORDER BY exit_time_utc DESC
                    LIMIT ?
                    """,
                    (since_utc, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM trades
                    WHERE exit_time_utc IS NOT NULL
                    ORDER BY exit_time_utc DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def list_trades_range(
        self,
        start_utc: str,
        end_utc: str,
        limit: int = 2000,
    ) -> List[Dict[str, Any]]:
        """List trades whose exit_time_utc is within [start_utc, end_utc)."""
        limit = int(limit)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT * FROM trades
                WHERE exit_time_utc IS NOT NULL
                  AND exit_time_utc >= ?
                  AND exit_time_utc < ?
                ORDER BY exit_time_utc DESC
                LIMIT ?
                """,
                (start_utc, end_utc, limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def delete_trades_range(self, start_utc: str, end_utc: str) -> int:
        """Delete trades whose exit_time_utc is within [start_utc, end_utc). Returns rows deleted."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                DELETE FROM trades
                WHERE exit_time_utc IS NOT NULL
                  AND exit_time_utc >= ?
                  AND exit_time_utc < ?
                """,
                (start_utc, end_utc),
            )
            deleted = cur.rowcount or 0
            self._conn.commit()
        return int(deleted)


