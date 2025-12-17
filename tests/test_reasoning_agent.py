import pandas as pd

from agents.reasoning_agent import ReasoningAgent
from agents.risk_agent import RiskAgent
from agents.subjectivity_agent import SubjectivityAgent


class StubStats:
    def __init__(self, score: float):
        self.score = score

    def predict(self, prices):
        return {"score": self.score}


class StubSubjectivity(SubjectivityAgent):
    def __init__(self, fixed_score: float):
        super().__init__()
        self.fixed_score = fixed_score

    def sentiment_score(self, text: str) -> float:  # pragma: no cover - deterministic override
        return self.fixed_score


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


def test_reasoning_bull_upweights_sentiment():
    prices = _price_series(trend=0.5)  # upward -> bull (sma20 > sma60)
    agent = ReasoningAgent(
        statistics=StubStats(score=0.2),
        risk=RiskAgent(rsi_cap=90, vol_cap=1.0),
        subjectivity=StubSubjectivity(fixed_score=1.0),
        reflection=None,
    )
    decision = agent.decide(prices, symbol="TEST", news_text="good news", prob_threshold=0.5)
    assert decision["action"] == "buy"
    assert decision["regime"] == "bull"


def test_reasoning_bear_downweights_sentiment():
    prices = _price_series(trend=-0.5)  # downward -> bear
    agent = ReasoningAgent(
        statistics=StubStats(score=0.3),
        risk=RiskAgent(rsi_cap=90, vol_cap=1.0),
        subjectivity=StubSubjectivity(fixed_score=-1.0),
        reflection=None,
    )
    decision = agent.decide(prices, symbol="TEST", news_text="bad news", prob_threshold=0.4)
    assert decision["action"] == "hold"
    assert decision["regime"] == "bear"
