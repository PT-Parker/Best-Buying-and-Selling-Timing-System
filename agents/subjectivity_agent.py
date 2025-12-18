from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SubjectivityAgent:
    """Deprecated: external-news sentiment is disabled in favor of expert reasoning."""

    llm_client: Any = None

    def analyze(self, news_text: str | None, ticker: str | None = None) -> Dict[str, Any]:
        return {
            "sentiment_score": 0.0,
            "reasoning": "外部新聞分析已停用",
            "ticker": ticker,
            "raw_text": news_text or "",
        }

    def sentiment_score(self, text: str) -> float:
        return 0.0
