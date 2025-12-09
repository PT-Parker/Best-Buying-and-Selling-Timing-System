import pandas as pd
import pytest

from core.backtest import StrategyConfig
from services import signals as signal_service
from services.data_source import DataSourceMode


@pytest.fixture
def recent_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": 100 + (dates.dayofyear % 5),
            "high": 101 + (dates.dayofyear % 5),
            "low": 99 + (dates.dayofyear % 5),
            "close": 100 + (dates.dayofyear % 4),
            "volume": 1_000_000,
            "symbol": "TEST.TW",
        }
    )
    return df


def test_summarize_signals_returns_rows(monkeypatch, recent_prices):
    def _loader(symbols, start, end, mode):
        return recent_prices

    monkeypatch.setattr(signal_service.data_source, "load_price_history", _loader)
    summary = signal_service.summarize_signals(
        symbols=["TEST.TW"],
        start="2024-01-01",
        end="2024-02-28",
        strategy=StrategyConfig(),
        mode=DataSourceMode.YFINANCE,
        model=None,
    )
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["symbol"] == "TEST.TW"
    assert "signal" in row and "confidence" in row
    assert summary["metadata"]["data_mode"] == DataSourceMode.YFINANCE.value


def test_summarize_signals_flags_anomalies(monkeypatch):
    def _loader(symbols, start, end, mode):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])

    monkeypatch.setattr(signal_service.data_source, "load_price_history", _loader)
    summary = signal_service.summarize_signals(
        symbols=["MISSING.TW"],
        start="2024-01-01",
        end="2024-02-01",
        strategy=StrategyConfig(),
        mode=DataSourceMode.YFINANCE,
        model=None,
    )
    assert summary["rows"] == []
    assert any("MISSING.TW" in msg for msg in summary["metadata"]["anomalies"])
