from __future__ import annotations

from dataclasses import dataclass

from core.features import FeatureConfig, build_features
from services import data_source

from .reasoning_agent import ReasoningAgent


@dataclass
class Orchestrator:
    reasoning: ReasoningAgent

    def run_decision(
        self,
        symbol: str,
        start: str,
        end: str,
        news_text: str = "",
        mode: data_source.DataSourceMode = data_source.DataSourceMode.YFINANCE,
        prob_threshold: float = 0.4,
    ) -> dict:
        prices = data_source.load_price_history([symbol], start, end, mode=mode)
        enriched = build_features(prices, FeatureConfig())
        decision = self.reasoning.decide(enriched, symbol, news_text, prob_threshold)
        return {
            "decision": decision,
            "prices": enriched,
        }
