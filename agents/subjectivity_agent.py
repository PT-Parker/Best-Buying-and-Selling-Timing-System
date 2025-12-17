from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SubjectivityAgent:
    """Extracts sentiment from unstructured text (placeholder implementation)."""

    positive_words: tuple = ("beat", "growth", "surge", "bull", "upgrade")
    negative_words: tuple = ("miss", "decline", "drop", "bear", "downgrade")

    PROMPT_TEMPLATE: str = (
        "Analyze sentiment (-1 to 1) for market text.\n"
        "Text:\n{text}\n"
        "Output: sentiment score only."
    )

    def sentiment_score(self, text: str) -> float:
        if not text:
            return 0.0
        lower = text.lower()
        pos_hits = sum(bool(re.search(rf"\\b{w}\\b", lower)) for w in self.positive_words)
        neg_hits = sum(bool(re.search(rf"\\b{w}\\b", lower)) for w in self.negative_words)
        score = pos_hits - neg_hits
        # Normalize to [-1, 1]
        if score == 0:
            return 0.0
        return max(-1.0, min(1.0, score / 3))
