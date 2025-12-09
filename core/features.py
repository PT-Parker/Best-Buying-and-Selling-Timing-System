from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


EMA_PREFIX = "ema_span_"
EMA_RATIO_PREFIX = "ema_ratio_"
RSI_PREFIX = "rsi_"
VOLUME_Z_PREFIX = "volume_zscore_"
RETURN_PREFIX = "return_"
VOLATILITY_PREFIX = "volatility_"

DEFAULT_FEATURES: Sequence[str] = (
    "ema_span_3",
    "ema_span_8",
    "ema_ratio_3_8",
    "ema_span_21",
    "ema_ratio_8_21",
    "rsi_7",
    "rsi_14",
    "volume_zscore_20",
    "return_1",
    "return_5",
    "volatility_5",
    "volatility_10",
)


@dataclass
class FeatureConfig:
    features: Sequence[str] = DEFAULT_FEATURES


def _grouped(prices: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    df = prices.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(["symbol", "date"], inplace=True)
    return df.groupby("symbol", group_keys=False)


def _compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def _compute_volume_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def build_features(prices: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    df = prices.copy()
    base_cols = {"open", "high", "low", "close", "volume"}
    grouped = _grouped(df)

    for feature in cfg.features:
        if feature in base_cols:
            # 原始欄位保持即可，不需計算
            continue
        if feature.startswith(EMA_PREFIX):
            span = int(feature.replace(EMA_PREFIX, ""))
            df[feature] = grouped["close"].transform(lambda s: _compute_ema(s, span))
        elif feature.startswith(EMA_RATIO_PREFIX):
            spans = feature.replace(EMA_RATIO_PREFIX, "").split("_")
            fast, slow = int(spans[0]), int(spans[1])
            fast_col = f"{EMA_PREFIX}{fast}"
            slow_col = f"{EMA_PREFIX}{slow}"
            if fast_col not in df:
                df[fast_col] = grouped["close"].transform(lambda s: _compute_ema(s, fast))
            if slow_col not in df:
                df[slow_col] = grouped["close"].transform(lambda s: _compute_ema(s, slow))
            df[feature] = df[fast_col] / df[slow_col] - 1
        elif feature.startswith(RSI_PREFIX):
            period = int(feature.replace(RSI_PREFIX, ""))
            df[feature] = grouped["close"].transform(lambda s: _compute_rsi(s, period))
        elif feature.startswith(VOLUME_Z_PREFIX):
            window = int(feature.replace(VOLUME_Z_PREFIX, ""))
            df[feature] = grouped["volume"].transform(lambda s: _compute_volume_zscore(s, window))
        elif feature.startswith(RETURN_PREFIX):
            window = int(feature.replace(RETURN_PREFIX, ""))
            df[feature] = grouped["close"].transform(lambda s: s.pct_change(window))
        elif feature.startswith(VOLATILITY_PREFIX):
            window = int(feature.replace(VOLATILITY_PREFIX, ""))
            df[feature] = grouped["close"].transform(lambda s: s.pct_change().rolling(window).std())
        else:
            raise ValueError(f"Unsupported feature definition: {feature}")

    return df
