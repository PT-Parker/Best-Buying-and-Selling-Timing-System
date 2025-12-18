import pandas as pd

from agents.reasoning_agent import ReasoningAgent
from agents.risk_agent import RiskAgent


class StubStats:
    def __init__(self, score: float):
        self.score = score

    def predict(self, prices):
        return {"score": self.score}


class FakeLLM:
    def __init__(self, scores, decision):
        self.scores = scores
        self.decision = decision
        self.calls = 0

    def chat(self, prompt: str, json_mode: bool = True):  # pragma: no cover - deterministic stub
        self.calls += 1
        return self.scores if self.calls == 1 else self.decision


def _price_series(trend: float) -> pd.DataFrame:
    base = 100.0
    closes = [base + i * trend for i in range(60)]
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "close": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "rsi_14": [50] * 60,
        }
    )
    return df


def test_reasoning_selects_active_role():
    prices = _price_series(trend=0.5)
    llm = FakeLLM(
        scores={"bull_score": 80, "bear_score": 10, "neutral_score": 20, "reasoning": "trend strong"},
        decision={"signal": "BUY", "confidence": 0.7, "reasoning": "breakout", "active_role": "bull"},
    )
    agent = ReasoningAgent(
        statistics=StubStats(score=0.3),
        risk=RiskAgent(rsi_cap=90, vol_cap=1.0),
        reflection=None,
        llm_client=llm,
    )
    decision = agent.decide(prices, symbol="TEST", time_summary="過去5日價格緩步上行。")
    assert decision["active_role"] == "bull"
    assert decision["action"] == "buy"
    assert decision["confidence"] == 0.7
    assert decision["score_reasoning"] == "trend strong"


def test_reasoning_risk_override():
    class BlockRisk(RiskAgent):
        def approve_trade(self, signal, prices):
            return False, "blocked"

    prices = _price_series(trend=0.5)
    llm = FakeLLM(
        scores={"bull_score": 70, "bear_score": 20, "neutral_score": 10, "reasoning": "momentum"},
        decision={"signal": "BUY", "confidence": 0.8, "reasoning": "breakout", "active_role": "bull"},
    )
    agent = ReasoningAgent(
        statistics=StubStats(score=0.4),
        risk=BlockRisk(rsi_cap=90, vol_cap=1.0),
        reflection=None,
        llm_client=llm,
    )
    decision = agent.decide(prices, symbol="TEST", time_summary="過去5日價格緩步上行。")
    assert decision["action"] == "hold"
