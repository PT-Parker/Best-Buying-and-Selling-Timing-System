from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk_agent import RiskAgent
from .statistics_agent import StatisticsAgent
from .subjectivity_agent import SubjectivityAgent
from .reflection_agent import ReflectionAgent


@dataclass
class ReasoningAgent:
    statistics: StatisticsAgent
    risk: RiskAgent
    subjectivity: SubjectivityAgent
    reflection: ReflectionAgent | None = None

    def _market_regime(self, prices: pd.DataFrame) -> str:
        if prices.empty or "close" not in prices:
            return "unknown"
        close = prices["close"]
        if len(close) < 60:
            return "unknown"
        sma20 = close.rolling(20).mean().iloc[-1]
        sma60 = close.rolling(60).mean().iloc[-1]
        return "bull" if sma20 > sma60 else "bear"

    def decide(
        self,
        prices: pd.DataFrame,
        symbol: str,
        news_text: str = "",
        prob_threshold: float = 0.4,
        market_regime: str | None = None,
    ) -> dict:
        stat_signal = self.statistics.predict(prices)
        subj = self.subjectivity.analyze(news_text, ticker=symbol)
        sentiment = subj.get("sentiment_score", 0.0)
        regime = market_regime or self._market_regime(prices)

        # Dynamic weighting aligned with spec: bear => stats >= 80%, bull => sentiment up to ~60%
        if regime == "bull":
            w_subj, w_stat = 0.55, 0.45
        elif regime == "bear":
            w_subj, w_stat = 0.15, 0.85
        else:
            w_subj, w_stat = 0.4, 0.6

        stat_score = stat_signal.get("score") or 0.0
        blended_score = w_stat * stat_score + w_subj * ((sentiment + 1) / 2)

        approve, risk_reason = self.risk.approve_trade(stat_signal, prices)
        guidelines = ""
        if self.reflection:
            guidelines = self.reflection.reflect_on_last_trade()

        action = "buy" if blended_score >= prob_threshold and approve else "hold"

        return {
            "symbol": symbol,
            "action": action,
            "blended_score": blended_score,
            "stat_score": stat_score,
            "sentiment": sentiment,
            "sentiment_reason": subj.get("reasoning", ""),
            "regime": regime,
            "approved": approve,
            "risk_reason": risk_reason,
            "guidelines": guidelines,
            "reason": f"regime={regime}, w_stat={w_stat}, w_subj={w_subj}, guidelines={guidelines or 'N/A'}",
        }
