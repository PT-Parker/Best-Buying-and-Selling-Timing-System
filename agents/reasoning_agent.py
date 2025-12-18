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
            raise RuntimeError("Gemini client is required for expert reasoning.")
        try:
            response = self.llm_client.chat(prompt, json_mode=True)
            if not response:
                raise ValueError("Gemini returned empty response.")
            return json.loads(response) if isinstance(response, str) else response
        except Exception:
            raise

    def _market_snapshot(self, prices: pd.DataFrame, stat_score: float, symbol: str) -> Dict[str, Any]:
        if prices.empty:
            return {"symbol": symbol, "stat_score": stat_score}
        close = prices["close"].astype(float)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal

        sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else float("nan")
        rsi = prices["rsi_14"].iloc[-1] if "rsi_14" in prices else float("nan")
        bb_std = close.rolling(20).std().iloc[-1] if len(close) >= 20 else float("nan")
        bb_up = sma20 + 2 * bb_std if np.isfinite(sma20) and np.isfinite(bb_std) else float("nan")
        bb_lo = sma20 - 2 * bb_std if np.isfinite(sma20) and np.isfinite(bb_std) else float("nan")
        vol10 = close.pct_change().rolling(10).std().iloc[-1] if len(close) >= 10 else float("nan")
        volume = prices["volume"].iloc[-1] if "volume" in prices else float("nan")
        volume_ma20 = prices["volume"].rolling(20).mean().iloc[-1] if "volume" in prices and len(prices) >= 20 else float("nan")
        atr = None
        if {"high", "low", "close"}.issubset(prices.columns):
            high = prices["high"].astype(float)
            low = prices["low"].astype(float)
            prev_close = close.shift()
            tr = np.maximum(
                high - low,
                np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
            )
            atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else float("nan")

        return {
            "symbol": symbol,
            "latest_close": float(close.iloc[-1]),
            "rsi_14": float(rsi) if np.isfinite(rsi) else None,
            "macd": float(macd.iloc[-1]) if len(macd) else None,
            "macd_signal": float(signal.iloc[-1]) if len(signal) else None,
            "macd_hist": float(macd_hist.iloc[-1]) if len(macd_hist) else None,
            "bb_upper": float(bb_up) if np.isfinite(bb_up) else None,
            "bb_lower": float(bb_lo) if np.isfinite(bb_lo) else None,
            "bb_width": float(bb_up - bb_lo) if np.isfinite(bb_up) and np.isfinite(bb_lo) else None,
            "volatility_10": float(vol10) if np.isfinite(vol10) else None,
            "atr_14": float(atr) if atr is not None and np.isfinite(atr) else None,
            "volume": float(volume) if np.isfinite(volume) else None,
            "volume_ma20": float(volume_ma20) if np.isfinite(volume_ma20) else None,
            "recent_closes": close.tail(10).round(4).tolist(),
            "stat_score": float(stat_score),
        }

    def _score_experts(self, market_data: Dict[str, Any], stat_score: float) -> Dict[str, int]:
        prompt = EXPERT_SCORING_PROMPT.format(
            market_data=json.dumps(market_data, ensure_ascii=False),
            model_score=f"{stat_score:.4f}",
        )
        llm_scores = self._call_llm(prompt)
        if not llm_scores:
            raise ValueError("Gemini scoring returned no data.")
        if not {"bull_score", "bear_score", "neutral_score"}.issubset(llm_scores.keys()):
            raise ValueError(f"Invalid scoring JSON: {llm_scores}")
        return {
            "bull_score": int(llm_scores["bull_score"]),
            "bear_score": int(llm_scores["bear_score"]),
            "neutral_score": int(llm_scores["neutral_score"]),
        }

    def _pick_active_role(self, scores: Dict[str, int]) -> str:
        return max(scores.items(), key=lambda kv: kv[1])[0].replace("_score", "")

    def decide(self, market_data: pd.DataFrame, symbol: str) -> dict:
        stat_signal = self.statistics.predict(market_data)
        stat_score = float(stat_signal.get("score") or 0.0)
        approve, risk_reason = self.risk.approve_trade(stat_signal, market_data)
        guidelines = self.reflection.reflect_on_last_trade() if self.reflection else ""

        snapshot = self._market_snapshot(market_data, stat_score, symbol)
        scores = self._score_experts(snapshot, stat_score)
        active_role = self._pick_active_role(scores)

        decision_prompt = EXPERT_DECISION_PROMPT.format(
            active_role=active_role,
            market_data=json.dumps(snapshot, ensure_ascii=False),
            model_score=f"{stat_score:.4f}",
            guidelines=guidelines or "N/A",
        )
        llm_decision = self._call_llm(decision_prompt)
        if not llm_decision:
            raise ValueError("Gemini decision returned no data.")

        signal = str(llm_decision.get("signal", "")).upper()
        confidence = llm_decision.get("confidence")
        reasoning = llm_decision.get("reasoning") or ""
        if signal not in {"BUY", "SELL", "HOLD"}:
            raise ValueError(f"Invalid decision JSON: {llm_decision}")
        if confidence is None:
            raise ValueError(f"Missing confidence in decision JSON: {llm_decision}")

        if not approve:
            signal = "HOLD"
            reasoning = f"風控攔截: {risk_reason}. {reasoning}".strip()
            confidence = min(float(confidence), 0.4)

        action = signal.lower()
        return {
            "symbol": symbol,
            "signal": signal,
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
