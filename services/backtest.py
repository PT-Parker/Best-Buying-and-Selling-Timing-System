from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from core import backtest as core_backtest
from core.backtest import BacktestResult, FeesConfig, StrategyConfig
from core.labeling import LabelConfig

from . import data_source


@dataclass
class BacktestRequest:
    symbol: str
    start: str
    end: str
    strategy: StrategyConfig
    fees: FeesConfig
    mode: data_source.DataSourceMode = data_source.DataSourceMode.AUTO


def _to_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    frame.set_index("Date", inplace=True)
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def run_backtest(request: BacktestRequest) -> BacktestResult:
    prices = data_source.load_price_history([request.symbol], request.start, request.end, mode=request.mode)
    symbol_prices = prices[prices["symbol"] == request.symbol]
    if symbol_prices.empty:
        raise ValueError(f"No price data for {request.symbol}")
    frame = _to_price_frame(symbol_prices)
    return core_backtest.run_backtest(frame, request.strategy, fees=request.fees, symbol=request.symbol)


def run_parameter_grid(
    symbols: Sequence[str],
    start: str,
    end: str,
    base_strategy: StrategyConfig,
    parameter_grid: Dict[str, Sequence],
    fees: FeesConfig,
    mode: data_source.DataSourceMode = data_source.DataSourceMode.AUTO,
) -> List[Dict]:
    rows: List[Dict] = []
    grid_keys = list(parameter_grid.keys())
    for values in itertools.product(*(parameter_grid[k] for k in grid_keys)):
        overrides = dict(zip(grid_keys, values))
        for sym in symbols:
            strategy = _apply_overrides(base_strategy, overrides)
            result = run_backtest(
                BacktestRequest(
                    symbol=sym,
                    start=start,
                    end=end,
                    strategy=strategy,
                    fees=fees,
                    mode=mode,
                )
            )
            row = {
                "symbol": sym,
                **overrides,
                **result.metrics,
            }
            rows.append(row)
    return rows


def _apply_overrides(strategy: StrategyConfig, overrides: Dict[str, float]) -> StrategyConfig:
    data = strategy.__dict__.copy()
    data.update(overrides)
    return StrategyConfig(**data)


def model_backtest_from_predictions(
    features: pd.DataFrame,
    predictions: Sequence[int],
    label_cfg: LabelConfig,
    fees: FeesConfig,
    symbol: str,
) -> BacktestResult:
    df = features.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["prediction"] = list(predictions)
    df["date"] = pd.to_datetime(df["date"])
    cash = fees.initial_capital
    shares = 0
    planned_exit: int | None = None
    entry_price = 0.0
    entry_date = None
    trades: List[Dict] = []
    equity_curve: List[float] = []

    tp = float(label_cfg.take_profit_pct)
    sl = abs(float(label_cfg.stop_loss_pct))

    for idx, row in df.iterrows():
        price = float(row["close"])

        # Exit conditions: take profit / stop loss / horizon / rollover signal
        if shares > 0:
            ret = (price - entry_price) / entry_price
            if (
                ret >= tp
                or ret <= -sl
                or (planned_exit is not None and idx >= planned_exit)
                or row["prediction"] == 1
            ):
                sell_px = price * (1 - fees.slippage)
                proceeds = shares * sell_px
                fee = proceeds * fees.commission
                cash += proceeds - fee
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": row["date"],
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(sell_px, 4),
                        "shares": shares,
                        "holding_days": (row["date"] - entry_date).days if entry_date is not None else None,
                        "pnl": round((sell_px - entry_price) * shares - fee, 2),
                    }
                )
                shares = 0
                entry_price = 0.0
                entry_date = None
                planned_exit = None

        # Entry after exit (or if flat)
        if shares == 0 and row["prediction"] == 1:
            buy_px = price * (1 + fees.slippage)
            qty = int((cash * (1 - fees.commission)) // buy_px)
            if qty > 0:
                cost = qty * buy_px
                fee = cost * fees.commission
                cash -= cost + fee
                shares = qty
                entry_price = buy_px
                entry_date = row["date"]
                planned_exit = min(idx + label_cfg.horizon_days, len(df) - 1)

        equity_curve.append(cash + shares * price)

    equity_df = pd.DataFrame({"Date": df["date"], "equity": equity_curve}).set_index("Date")
    trades_df = pd.DataFrame(trades)
    metrics = core_backtest._compute_metrics(  # type: ignore[attr-defined]
        pd.DataFrame({"equity": equity_curve, "signal": df["prediction"]}),
        trades_df,
        fees,
    )
    return BacktestResult(
        symbol=symbol,
        metrics=metrics,
        trades=trades_df,
        equity_curve=equity_df,
        signals=df,
    )
