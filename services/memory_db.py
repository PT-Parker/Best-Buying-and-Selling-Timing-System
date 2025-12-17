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
    input_features TEXT,
    model_prediction TEXT,
    actual_outcome TEXT,
    strategy_reasoning TEXT
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

    def insert_trade(
        self,
        input_features: Dict[str, Any] | None,
        model_prediction: str,
        actual_outcome: str,
        strategy_reasoning: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_history (timestamp, input_features, model_prediction, actual_outcome, strategy_reasoning)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.datetime.utcnow().isoformat() + "Z",
                    json.dumps(input_features or {}),
                    model_prediction,
                    actual_outcome,
                    strategy_reasoning,
                ),
            )

    def fetch_last_trade(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
