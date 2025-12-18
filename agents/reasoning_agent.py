from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict

import numpy as np
import pandas as pd

from .risk_agent import RiskAgent
from .statistics_agent import StatisticsAgent
from .reflection_agent import ReflectionAgent
from .prompts import EXPERT_SCORING_PROMPT, EXPERT_DECISION_PROMPT


@dataclass
class ReasoningAgent:
    statistics: StatisticsAgent
    risk: RiskAgent
    reflection: ReflectionAgent | None = None
    llm_client: Any = None

    def _call_llm(self, prompt: str) -> Dict[str, Any] | None:
        if not self.llm_client:
            return None
        try:
            if callable(self.llm_client):
                response = self.llm_client(prompt)
            elif hasattr(self.llm_client, "chat"):
                response = self.llm_client.chat(prompt)
            else:
                return None
            if not response:
                return None
            return json.loads(response) if isinstance(response, str) else response
        except Exception:
            return None

    def _market_snapshot(self, prices: pd.DataFrame, stat_score: float, symbol: str) -> Dict[str, Any]:
        if prices.empty:
            return {"symbol": symbol, "stat_score": stat_score}
        close = prices["close"].astype(float)
        sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else float("nan")
        sma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else float("nan")
        rsi = prices["rsi_14"].iloc[-1] if "rsi_14" in prices else float("nan")
        vol10 = close.pct_change().rolling(10).std().iloc[-1] if len(close) >= 10 else float("nan")
        bb_mid = sma20
        bb_std = close.rolling(20).std().iloc[-1] if len(close) >= 20 else float("nan")
        bb_up = bb_mid + 2 * bb_std if np.isfinite(bb_mid) and np.isfinite(bb_std) else float("nan")
        bb_lo = bb_mid - 2 * bb_std if np.isfinite(bb_mid) and np.isfinite(bb_std) else float("nan")
        return {
            "symbol": symbol,
            "latest_close": float(close.iloc[-1]),
            "sma20": float(sma20) if np.isfinite(sma20) else None,
            "sma60": float(sma60) if np.isfinite(sma60) else None,
            "rsi_14": float(rsi) if np.isfinite(rsi) else None,
            "bb_upper": float(bb_up) if np.isfinite(bb_up) else None,
            "bb_lower": float(bb_lo) if np.isfinite(bb_lo) else None,
            "volatility_10": float(vol10) if np.isfinite(vol10) else None,
            "recent_closes": close.tail(5).round(4).tolist(),
            "stat_score": float(stat_score),
        }

    def _score_experts(self, market_data: Dict[str, Any], stat_score: float) -> Dict[str, int]:
        prompt = EXPERT_SCORING_PROMPT.format(
            market_data=json.dumps(market_data, ensure_ascii=False),
            model_score=f"{stat_score:.4f}",
        )
        llm_scores = self._call_llm(prompt) or {}
        if {"bull_score", "bear_score", "neutral_score"}.issubset(llm_scores.keys()):
            try:
                return {
                    "bull_score": int(llm_scores["bull_score"]),
                    "bear_score": int(llm_scores["bear_score"]),
                    "neutral_score": int(llm_scores["neutral_score"]),
                }
            except Exception:
                pass

        # Heuristic fallback when LLM is unavailable
        bull = 50
        bear = 50
        neutral = 50
        sma20 = market_data.get("sma20")
        sma60 = market_data.get("sma60")
        rsi = market_data.get("rsi_14")
        vol = market_data.get("volatility_10")
        if sma20 is not None and sma60 is not None:
            if sma20 > sma60:
                bull += 25
                bear -= 15
            else:
                bear += 25
                bull -= 15
        if rsi is not None:
            if rsi > 70:
                bear += 10
                bull -= 10
            elif rsi < 30:
                bull += 10
        if vol is not None and vol > 0.02:
            neutral += 15

        return {
            "bull_score": int(max(0, min(100, bull))),
            "bear_score": int(max(0, min(100, bear))),
            "neutral_score": int(max(0, min(100, neutral))),
        }

    def _pick_active_role(self, scores: Dict[str, int]) -> str:
        return max(scores.items(), key=lambda kv: kv[1])[0].replace("_score", "")

    def decide(self, prices: pd.DataFrame, symbol: str, prob_threshold: float = 0.4) -> dict:
        stat_signal = self.statistics.predict(prices)
        stat_score = float(stat_signal.get("score") or 0.0)
        approve, risk_reason = self.risk.approve_trade(stat_signal, prices)
        guidelines = self.reflection.reflect_on_last_trade() if self.reflection else ""

        market_data = self._market_snapshot(prices, stat_score, symbol)
        scores = self._score_experts(market_data, stat_score)
        active_role = self._pick_active_role(scores)

        decision_prompt = EXPERT_DECISION_PROMPT.format(
            active_role=active_role,
            market_data=json.dumps(market_data, ensure_ascii=False),
            model_score=f"{stat_score:.4f}",
            guidelines=guidelines or "N/A",
        )
        llm_decision = self._call_llm(decision_prompt) or {}

        action = str(llm_decision.get("action", "")).lower()
        confidence = llm_decision.get("confidence")
        reasoning = llm_decision.get("reasoning") or ""
        if action not in {"buy", "sell", "hold"}:
            action = "buy" if active_role == "bull" else ("sell" if active_role == "bear" else "hold")
        if confidence is None:
            confidence = max(scores.values()) / 100.0

        if not approve:
            action = "hold"
            reasoning = f"風控攔截: {risk_reason}. {reasoning}".strip()
            confidence = min(float(confidence), 0.4)

        return {
            "symbol": symbol,
            "action": action,
            "confidence": float(confidence),
            "reasoning": reasoning or f"active_role={active_role}, scores={scores}",
            "active_role": active_role,
            "expert_scores": scores,
            "stat_score": stat_score,
            "approved": approve,
            "risk_reason": risk_reason,
            "guidelines": guidelines,
        }
