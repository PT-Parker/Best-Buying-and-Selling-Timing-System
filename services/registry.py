from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_REGISTRY = Path("metrics/model_registry.jsonl")


def append(entry: Dict, path: Path = DEFAULT_REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False))
        fh.write("\n")


def load_all(path: Path = DEFAULT_REGISTRY) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def latest(path: Path = DEFAULT_REGISTRY, symbol: str | None = None) -> Optional[Dict]:
    entries = load_all(path)
    if not entries:
        return None
    if symbol is None:
        return entries[-1]

    symbol_norm = symbol.lower()
    for entry in reversed(entries):
        entry_symbol = str(entry.get("symbol", "")).lower()
        if entry_symbol == symbol_norm:
            return entry
    return None
