from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from core import inference as core_inference
from core.features import FeatureConfig, build_features


@dataclass
class StatisticsAgent:
    """Wrapper around XGBoost inference to produce probabilistic signals."""

    booster: any  # xgboost.Booster
    feature_columns: Sequence[str]

    def predict(self, prices: pd.DataFrame) -> dict:
        if prices.empty:
            raise ValueError("No price data supplied to StatisticsAgent")
        feat_cfg = FeatureConfig(features=list(self.feature_columns))
        feats = build_features(prices, feat_cfg)
        feats = feats.dropna(subset=self.feature_columns)
        if feats.empty:
            raise ValueError("Insufficient features for prediction")
        score = core_inference.score_features(feats.tail(1), self.booster, self.feature_columns)
        return {
            "score": score,
            "as_of": feats["date"].max() if "date" in feats else None,
            "features": feats.tail(1),
        }
