from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import yaml

try:  # pragma: no cover
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover
    yf = None

from core.backtest import FeesConfig, StrategyConfig
from core.labeling import LabelConfig


class DataSourceMode(str, Enum):
    AUTO = "auto"
    YFINANCE = "yfinance"


WATCHLIST_PATH = Path(os.environ.get("WATCHLIST_PATH", "config/watchlist.yaml"))
STRATEGY_PATH = Path(os.environ.get("STRATEGY_PATH", "config/strategy.yaml"))


@dataclass
class Watchlist:
    symbols: List[str]
    timezone: str = "Asia/Taipei"


@lru_cache(maxsize=4)
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_watchlist(path: Path = WATCHLIST_PATH, fallback: Path = Path("data/sample/watchlist.yaml")) -> Watchlist:
    data = _load_yaml(path) or _load_yaml(fallback)
    symbols = data.get("symbols") or ["2330.TW"]
    tz = data.get("timezone", "Asia/Taipei")
    return Watchlist(symbols=list(symbols), timezone=tz)


def load_strategy_config(path: Path = STRATEGY_PATH) -> tuple[StrategyConfig, FeesConfig]:
    data = _load_yaml(path)
    strategy = data.get("strategy") or {}
    fees_data = data.get("fees") or {}
    strat_cfg = StrategyConfig(
        ema_fast=int(strategy.get("ema_fast", 5)),
        ema_slow=int(strategy.get("ema_slow", 20)),
        rsi_period=int(strategy.get("rsi_period", 14)),
        rsi_buy_lt=float(strategy.get("rsi_buy_lt", 30)),
        rsi_sell_gt=float(strategy.get("rsi_sell_gt", 70)),
        bollinger_period=int(strategy.get("bollinger_period", 20)),
        bollinger_std=float(strategy.get("bollinger_std", 2.0)),
        take_profit_pct=float(strategy.get("take_profit_pct", 0.03)),
        stop_loss_pct=float(strategy.get("stop_loss_pct", 0.015)),
        cooldown_days=int(strategy.get("cooldown_days", 0)),
    )
    fees = FeesConfig(
        commission=float(fees_data.get("commission", 0.001425)),
        slippage=float(fees_data.get("slippage", 0.0005)),
        initial_capital=float(fees_data.get("initial_capital", 1_000_000.0)),
    )
    return strat_cfg, fees


def load_label_config(path: Path = STRATEGY_PATH) -> LabelConfig:
    data = _load_yaml(path)
    lbl = data.get("labeling") or {}
    return LabelConfig(
        horizon_days=int(lbl.get("horizon_days", LabelConfig().horizon_days)),
        take_profit_pct=float(lbl.get("take_profit_pct", LabelConfig().take_profit_pct)),
        stop_loss_pct=float(lbl.get("stop_loss_pct", LabelConfig().stop_loss_pct)),
    )


def _download_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed; cannot download prices")
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [col.split("_")[0] if "_" in col else col for col in df.columns]
    df = df.rename_axis("date").reset_index()
    df.rename(columns=str.lower, inplace=True)
    df["symbol"] = symbol
    return df


def load_price_history(
    symbols: Iterable[str],
    start: str,
    end: str,
    mode: DataSourceMode | None = None,
) -> pd.DataFrame:
    data_mode = mode or DataSourceMode.AUTO
    if data_mode not in (DataSourceMode.AUTO, DataSourceMode.YFINANCE):
        raise ValueError("Only yfinance data source is supported.")

    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        fetched = _download_prices(symbol, start, end)
        if fetched.empty:
            raise ValueError(f"No price data fetched for {symbol} in range {start}~{end}")
        frames.append(fetched)
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined.sort_values(["symbol", "date"], inplace=True)
    return combined.reset_index(drop=True)
