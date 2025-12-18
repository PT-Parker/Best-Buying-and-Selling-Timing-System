import json

from services.memory_db import MemoryDB


def test_memory_db_inserts_explanation(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    db = MemoryDB(path=db_path)
    payload = {"symbol": "TEST", "score": 0.62}
    db.insert_explanation(
        symbol="TEST",
        context="decision_card",
        payload=payload,
        explanation="Line one\nLine two",
    )
    rows = db.get_recent_explanations(limit=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "TEST"
    assert row["context"] == "decision_card"
    assert json.loads(row["payload"]) == payload
    assert "Line one" in row["explanation"]
