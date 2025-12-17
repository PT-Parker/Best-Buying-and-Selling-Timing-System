from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional
import os

from .prompts import SENTIMENT_PROMPT

try:  # pragma: no cover - optional dependency
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


@dataclass
class SubjectivityAgent:
    """LLM-backed sentiment extraction with heuristic fallback."""

    llm_client: Any = None
    positive_words: tuple = ("beat", "growth", "surge", "bull", "upgrade")
    negative_words: tuple = ("miss", "decline", "drop", "bear", "downgrade")
    ticker_aliases: Dict[str, List[str]] = None

    def __post_init__(self):
        self.ticker_aliases = self.ticker_aliases or {
            "2330.TW": ["台積電", "TSMC", "2330"],
            "0050.TW": ["台灣50", "0050", "台灣大盤ETF"],
            "BTC": ["bitcoin", "BTC"],
            "ETH": ["ethereum", "ETH"],
        }

    def fetch_headlines(self, ticker: str | None = None, limit: int = 5) -> str:
        """Fetch headlines from CryptoPanic or GNews if configured; fallback to mock text."""
        aliases = self.ticker_aliases.get(ticker or "", []) if self.ticker_aliases else []
        queries = [q for q in [ticker, *aliases, "crypto", "market"] if q]

        # Try CryptoPanic if token present
        cp_token = os.getenv("CRYPTOPANIC_API_KEY")
        if cp_token and requests:
            params = {"auth_token": cp_token, "public": "true"}
            if ticker:
                params["currencies"] = ticker
            try:  # pragma: no cover - network path
                resp = requests.get("https://cryptopanic.com/api/v1/posts/", params=params, timeout=5)
                if resp.ok:
                    data = resp.json()
                    headlines = [
                        (item.get("title") or item.get("slug") or "") for item in data.get("results", []) if item.get("title") or item.get("slug")
                    ]
                    if headlines:
                        return " ".join(headlines[:limit])
            except Exception:
                pass

        # Try GNews if token present
        token = os.getenv("GNEWS_API_KEY")
        if token and requests:
            # Combine aliases to widen recall
            q = " OR ".join(queries[:3]) if queries else "market"
            try:  # pragma: no cover - network path
                resp = requests.get(
                    "https://gnews.io/api/v4/search",
                    params={"q": q, "lang": "zh,en", "max": limit, "token": token},
                    timeout=5,
                )
                if resp.ok:
                    data = resp.json()
                    headlines = [a.get("title", "") for a in data.get("articles", []) if a.get("title")]
                    if headlines:
                        return " ".join(headlines)
            except Exception:
                pass

        # Fallback mock content
        mock_news = [
            f"Analyst predicts {ticker or 'asset'} may surge on upgrade",
            f"Market watches resistance as {ticker or 'asset'} consolidates",
        ]
        return " ".join(mock_news[:limit])

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

    def analyze(self, news_text: str | None, ticker: str | None = None) -> Dict[str, Any]:
        text = news_text or self.fetch_headlines(ticker=ticker)
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
