from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .prompts import SENTIMENT_PROMPT


@dataclass
class SubjectivityAgent:
    """LLM-backed sentiment extraction with heuristic fallback."""

    llm_client: Any = None
    positive_words: tuple = ("beat", "growth", "surge", "bull", "upgrade")
    negative_words: tuple = ("miss", "decline", "drop", "bear", "downgrade")

    def _call_llm(self, news_text: str) -> Tuple[float, str] | None:
        if not self.llm_client:
            return None
        prompt = SENTIMENT_PROMPT.format(news_text=news_text)
        try:
            # support either callable or .chat interface
            if callable(self.llm_client):
                response = self.llm_client(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = self.llm_client.chat(prompt)
            else:
                return None
            if not response:
                return None
            parsed = json.loads(response) if isinstance(response, str) else response
            score = float(parsed.get("sentiment_score", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            return max(-1.0, min(1.0, score)), reasoning
        except Exception:
            return None

    def _heuristic(self, text: str) -> Tuple[float, str]:
        lower = text.lower()
        pos_hits = sum(bool(re.search(rf"\\b{w}\\b", lower)) for w in self.positive_words)
        neg_hits = sum(bool(re.search(rf"\\b{w}\\b", lower)) for w in self.negative_words)
        score = pos_hits - neg_hits
        if score == 0:
            return 0.0, "中立或無明顯正負關鍵字"
        normalized = max(-1.0, min(1.0, score / 3))
        reason = "正面關鍵字多" if normalized > 0 else "負面關鍵字多"
        return normalized, reason

    def analyze(self, news_text: str, ticker: str | None = None) -> Dict[str, Any]:
        text = news_text or ""
        llm_res = self._call_llm(text)
        if llm_res:
            score, reason = llm_res
        else:
            score, reason = self._heuristic(text)
        return {
            "sentiment_score": score,
            "reasoning": reason,
            "ticker": ticker,
            "raw_text": text,
        }

    def sentiment_score(self, text: str) -> float:
        return self.analyze(text).get("sentiment_score", 0.0)
