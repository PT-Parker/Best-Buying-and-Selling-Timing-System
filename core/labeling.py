from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class LabelConfig:
    horizon_days: int = 2
    take_profit_pct: float = 0.01
    stop_loss_pct: float = -0.01


def compute_forward_returns(frame: pd.DataFrame, horizon: int) -> pd.Series:
    future_price = frame.groupby("symbol", group_keys=False)["close"].shift(-horizon)
    returns = future_price / frame["close"] - 1
    return returns


def assign_labels(frame: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
    cfg = config or LabelConfig()
    df = frame.copy()
    df["future_return"] = compute_forward_returns(df, cfg.horizon_days)
    df["label"] = 0
    df.loc[df["future_return"] >= cfg.take_profit_pct, "label"] = 1
    df.loc[df["future_return"] <= cfg.stop_loss_pct, "label"] = -1
    return df
