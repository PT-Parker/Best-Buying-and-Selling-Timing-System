from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence

import pandas as pd

try:  # pragma: no cover
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None

from core import backtest as core_backtest
from core import features as core_features
from core.features import FeatureConfig
from core import inference as core_inference

from . import data_source


@dataclass
class ModelContext:
    booster: Any
    feature_columns: Sequence[str]
    threshold: float = 0.65


def _to_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    frame.set_index("Date", inplace=True)
    return frame[["Open", "High", "Low", "Close", "Volume"]]


def summarize_signals(
    symbols: Sequence[str],
    start: str,
    end: str,
    strategy: core_backtest.StrategyConfig,
    mode: data_source.DataSourceMode = data_source.DataSourceMode.AUTO,
    model: ModelContext | None = None,
) -> Dict:
    summaries: List[Dict] = []
    anomalies: List[str] = []
    prices = data_source.load_price_history(symbols, start, end, mode=mode)
    for symbol in symbols:
        sym_prices = prices[prices["symbol"] == symbol]
        if sym_prices.empty:
            anomalies.append(f"{symbol}: 無資料")
            continue
        frame = _to_price_frame(sym_prices)
        signal_df = core_backtest.generate_signals(frame, strategy)
        latest = signal_df.dropna(subset=["Close"]).tail(1)
        if latest.empty:
            anomalies.append(f"{symbol}: 無足夠資料")
            continue
        row = latest.iloc[0]
        as_of = latest.index[0]
        score = None
        if model and xgb is not None:
            enriched = sym_prices.rename(columns=str.lower).copy()
            enriched["symbol"] = symbol
            enriched["date"] = pd.to_datetime(enriched["date"])
            feature_cfg = FeatureConfig(features=list(model.feature_columns))
            feats = core_features.build_features(enriched, feature_cfg)
            latest_feat = feats.tail(1)
            score = core_inference.score_features(latest_feat, model.booster, model.feature_columns)
            if score is not None and score < model.threshold:
                anomalies.append(f"{symbol}: 模型分數 {score:.2f} < 門檻 {model.threshold}")
        if (datetime.utcnow() - as_of.to_pydatetime()) > timedelta(days=3):
            anomalies.append(f"{symbol}: 價格時間 {as_of.date()} 過舊")
        summaries.append(
            {
                "symbol": symbol,
                "as_of": as_of.isoformat(),
                "close": float(row["Close"]),
                "signal": row["signal"],
                "confidence": row["confidence"],
                "reason": row["reason"],
                "score": score,
            }
        )

    summaries.sort(key=lambda x: (x["signal"] != "HOLD", x.get("score", 0) or 0), reverse=True)
    return {
        "rows": summaries,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "data_mode": mode.value if isinstance(mode, data_source.DataSourceMode) else mode,
            "anomalies": anomalies,
        },
    }
