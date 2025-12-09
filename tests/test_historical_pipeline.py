import pandas as pd

from core.backtest import FeesConfig, StrategyConfig, generate_signals, run_backtest


def _sample_price_frame(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    frame = pd.DataFrame(
        {
            "Open": 100 + (dates.dayofyear % 5),
            "High": 101 + (dates.dayofyear % 5),
            "Low": 99 + (dates.dayofyear % 5),
            "Close": 100 + (dates.dayofyear % 3),
            "Volume": 1_000_000,
        },
        index=dates,
    )
    return frame


def test_generate_signals_has_core_columns():
    frame = _sample_price_frame()
    enriched = generate_signals(frame, StrategyConfig())
    assert {"signal", "confidence", "RSI", "BB_UP", "BB_LO"}.issubset(enriched.columns)


def test_run_backtest_returns_metrics_and_trades():
    frame = _sample_price_frame()
    result = run_backtest(frame, StrategyConfig(), FeesConfig(), symbol="TEST.TW")
    assert result.metrics["signal_count"] >= 0
    assert isinstance(result.trades, pd.DataFrame)
    assert isinstance(result.equity_curve, pd.DataFrame)
