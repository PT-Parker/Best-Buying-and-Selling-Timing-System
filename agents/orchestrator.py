from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.features import FeatureConfig, build_features
from services import data_source, memory_db

from .reasoning_agent import ReasoningAgent


@dataclass
class Orchestrator:
    reasoning: ReasoningAgent
    db: memory_db.MemoryDB | None = None

    def _build_time_summary(self, prices: pd.DataFrame, window: int = 5) -> str:
        if prices.empty or "close" not in prices:
            return "近期資料不足，無法產生時間敘述。"

        recent = prices.sort_values("date").tail(window).copy()
        closes = recent["close"].astype(float)
        start = closes.iloc[0]
        end = closes.iloc[-1]
        if pd.isna(start) or pd.isna(end):
            return "近期資料不足，無法產生時間敘述。"

        high = closes.max()
        low = closes.min()
        change_pct = ((end - start) / start * 100) if start else 0.0
        range_pct = ((high - low) / start * 100) if start else 0.0

        summary = (
            f"過去{len(recent)}日價格由 {start:.2f} 走到 {end:.2f}（{change_pct:+.2f}%），"
            f"區間高點 {high:.2f}、低點 {low:.2f}，區間波動約 {range_pct:.2f}%。"
        )

        changes = closes.pct_change().dropna()
        if not changes.empty:
            max_move_idx = changes.abs().idxmax()
            max_move = changes.loc[max_move_idx] * 100
            move_date = recent.loc[max_move_idx, "date"]
            move_date_str = move_date.strftime("%Y-%m-%d") if hasattr(move_date, "strftime") else str(move_date)
            summary += f" 最大單日變動出現在 {move_date_str}，漲跌幅 {max_move:+.2f}%。"

        if "volume" in recent:
            volumes = recent["volume"].astype(float)
            avg_vol = float(volumes.mean())
            if avg_vol > 0:
                last_vol = float(volumes.iloc[-1])
                ratio = last_vol / avg_vol
                summary += f" 最新量能為近{len(recent)}日均量的 {ratio:.2f} 倍。"

        return summary

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
    ) -> dict:
        prices = data_source.load_price_history([symbol], start, end, mode=mode)
        enriched = build_features(prices, FeatureConfig())

        latest_price = float(enriched.sort_values("date")["close"].iloc[-1]) if not enriched.empty else None
        if latest_price is not None:
            self._update_previous_profit(latest_price)

        time_summary = self._build_time_summary(enriched)
        decision = self.reasoning.decide(enriched, symbol, time_summary=time_summary)

        if self.db is not None and latest_price is not None:
            trade_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "symbol": symbol,
                "signal": decision.get("signal") or decision.get("action"),
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
