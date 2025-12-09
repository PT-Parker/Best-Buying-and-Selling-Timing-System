from pathlib import Path

import pandas as pd
import pytest

from core.backtest import FeesConfig, StrategyConfig
from services import backtest as backtest_service


@pytest.fixture(scope="module")
def price_fixture() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "fixtures" / "sample_prices.csv"
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.rename(columns=str.lower, inplace=True)
    df["symbol"] = "TEST.TW"
    return df


def test_run_parameter_grid_returns_metrics(monkeypatch, price_fixture):
    def _fake_loader(symbols, start, end, mode):
        return price_fixture

    monkeypatch.setattr(backtest_service.data_source, "load_price_history", _fake_loader)

    rows = backtest_service.run_parameter_grid(
        symbols=["TEST.TW"],
        start="2024-01-01",
        end="2024-02-29",
        base_strategy=StrategyConfig(),
        parameter_grid={"ema_fast": [5, 8], "ema_slow": [12, 20]},
        fees=FeesConfig(),
    )

    assert len(rows) == 4
    assert {"symbol", "annual_return", "ema_fast", "ema_slow"}.issubset(rows[0].keys())
    assert all(row["symbol"] == "TEST.TW" for row in rows)


def test_run_parameter_grid_handles_multiple_symbols(monkeypatch, price_fixture):
    def _loader(symbols, start, end, mode):
        frames = []
        for sym in symbols:
            df = price_fixture.copy()
            df["symbol"] = sym
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    monkeypatch.setattr(backtest_service.data_source, "load_price_history", _loader)

    rows = backtest_service.run_parameter_grid(
        symbols=["AAA.TW", "BBB.TW"],
        start="2024-01-01",
        end="2024-02-29",
        base_strategy=StrategyConfig(),
        parameter_grid={"ema_fast": [5], "ema_slow": [20]},
        fees=FeesConfig(),
    )
    assert len(rows) == 2
    assert {row["symbol"] for row in rows} == {"AAA.TW", "BBB.TW"}
