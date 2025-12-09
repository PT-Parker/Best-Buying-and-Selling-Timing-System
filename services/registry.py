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


def latest(path: Path = DEFAULT_REGISTRY) -> Optional[Dict]:
    entries = load_all(path)
    return entries[-1] if entries else None
