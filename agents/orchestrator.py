from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.features import FeatureConfig, build_features
from services import data_source, memory_db

from .reasoning_agent import ReasoningAgent


@dataclass
class Orchestrator:
    reasoning: ReasoningAgent
    db: memory_db.MemoryDB | None = None

    def _update_previous_profit(self, latest_price: float) -> None:
        if self.db is None:
            return
        self.db.update_last_trade_profit(actual_price=latest_price)

    def run_decision(
        self,
        symbol: str,
        start: str,
        end: str,
        mode: data_source.DataSourceMode = data_source.DataSourceMode.YFINANCE,
        prob_threshold: float = 0.4,
    ) -> dict:
        prices = data_source.load_price_history([symbol], start, end, mode=mode)
        enriched = build_features(prices, FeatureConfig())

        latest_price = float(enriched.sort_values("date")["close"].iloc[-1]) if not enriched.empty else None
        if latest_price is not None:
            self._update_previous_profit(latest_price)

        decision = self.reasoning.decide(enriched, symbol, prob_threshold)

        if self.db is not None and latest_price is not None:
            trade_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "symbol": symbol,
                "signal": decision.get("action"),
                "reasoning": decision.get("reasoning"),
                "market_regime": decision.get("active_role"),
                "entry_price": latest_price,
                "stat_score": decision.get("stat_score"),
                "guidelines": decision.get("guidelines"),
                "confidence": decision.get("confidence"),
                "active_role": decision.get("active_role"),
                "expert_scores": decision.get("expert_scores"),
                "actual_profit": None,
                "actual_outcome": None,
            }
            self.db.insert_trade(trade_record)

        return {
            "decision": decision,
            "prices": enriched,
        }

    # Alias to match requested naming
    def run_analysis(self, *args, **kwargs):
        return self.run_decision(*args, **kwargs)
