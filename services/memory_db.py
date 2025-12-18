from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS trade_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT,
    signal TEXT,
    reasoning TEXT,
    market_regime TEXT,
    entry_price REAL,
    stat_score REAL,
    sentiment REAL,
    blended_score REAL,
    guidelines TEXT,
    active_role TEXT,
    confidence REAL,
    expert_scores TEXT,
    actual_profit REAL,
    actual_outcome TEXT
);
"""


class MemoryDB:
    def __init__(self, path: str | Path = "data/memory.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(trade_history)").fetchall()}
        expected = {
            "symbol": "TEXT",
            "signal": "TEXT",
            "reasoning": "TEXT",
            "market_regime": "TEXT",
            "entry_price": "REAL",
            "stat_score": "REAL",
            "sentiment": "REAL",
            "blended_score": "REAL",
            "guidelines": "TEXT",
            "active_role": "TEXT",
            "confidence": "REAL",
            "expert_scores": "TEXT",
            "actual_profit": "REAL",
            "actual_outcome": "TEXT",
        }
        for name, col_type in expected.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE trade_history ADD COLUMN {name} {col_type}")

    def insert_trade(self, trade_data: Dict[str, Any]) -> None:
        """Insert a new trade decision record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_history
                (timestamp, symbol, signal, reasoning, market_regime, entry_price, stat_score, sentiment, blended_score, guidelines, active_role, confidence, expert_scores, actual_profit, actual_outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_data.get("timestamp") or datetime.datetime.utcnow().isoformat() + "Z",
                    trade_data.get("symbol"),
                    trade_data.get("signal"),
                    trade_data.get("reasoning"),
                    trade_data.get("market_regime"),
                    trade_data.get("entry_price"),
                    trade_data.get("stat_score"),
                    trade_data.get("sentiment"),
                    trade_data.get("blended_score"),
                    trade_data.get("guidelines"),
                    trade_data.get("active_role"),
                    trade_data.get("confidence"),
                    json.dumps(trade_data.get("expert_scores") or {}),
                    trade_data.get("actual_profit"),
                    trade_data.get("actual_outcome"),
                ),
            )

    def update_last_trade_profit(self, actual_price: float) -> None:
        """Update the most recent trade with realized/mark-to-market profit if missing."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return
            if row["actual_profit"] is not None and row["actual_outcome"] is not None:
                return
            entry_price = row["entry_price"]
            signal = row["signal"]
            if entry_price is None or signal is None:
                return
            direction = 1 if signal == "buy" else (-1 if signal == "sell" else 0)
            profit = (actual_price - entry_price) * direction
            outcome = "up" if profit > 0 else ("down" if profit < 0 else "flat")
            conn.execute(
                "UPDATE trade_history SET actual_profit = ?, actual_outcome = ? WHERE id = ?",
                (profit, outcome, row["id"]),
            )

    def fetch_last_trade(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_recent_trades(self, limit: int = 10) -> list[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
