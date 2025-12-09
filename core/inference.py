from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency guard
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None

from sklearn.metrics import precision_recall_fscore_support

from . import labeling


@dataclass
class TrainingArtifacts:
    model_path: Path
    metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    feature_columns: Sequence[str]
    model: Optional["xgb.XGBClassifier"]


def _select_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, Sequence[str]]:
    exclude = {"symbol", "date", "future_return", "label"}
    cols = [c for c in frame.columns if c not in exclude]
    if not cols:
        raise ValueError("No feature columns available for training")
    return frame[cols], cols


def train_classifier(
    feature_frame: pd.DataFrame,
    label_config: labeling.LabelConfig | None = None,
    hyperparameters: Optional[dict] = None,
    output_path: Optional[str | Path] = None,
) -> TrainingArtifacts:
    if xgb is None:
        raise RuntimeError("xgboost is required for training classifiers")
    cfg = label_config or labeling.LabelConfig()
    df = labeling.assign_labels(feature_frame, cfg)
    df = df.dropna(subset=["label"])
    df = df[df["label"] != 0]
    df.loc[df["label"] == -1, "label"] = 0
    df.sort_values(["symbol", "date"], inplace=True)
    if df.empty:
        raise ValueError("No labelled rows after applying thresholds")

    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff]
    test_df = df.iloc[cutoff:]

    X_train, feature_cols = _select_features(train_df)
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
    }
    if hyperparameters:
        params.update(hyperparameters)

    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    if pos_count > 0:
        params.setdefault("scale_pos_weight", max(1.0, neg_count / max(1, pos_count)))

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_future_return": float(test_df["future_return"].mean()),
    }
    feature_importance = model.get_booster().get_score(importance_type="gain")

    output = Path(output_path) if output_path else Path("models") / "model_latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output)
    return TrainingArtifacts(
        model_path=output,
        metrics=metrics,
        feature_importance=feature_importance,
        feature_columns=list(feature_cols),
        model=model,
    )


def load_booster(model_path: str | Path) -> xgb.Booster:
    if xgb is None:
        raise RuntimeError("xgboost is not installed; cannot load model")
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def score_features(
    features_df: pd.DataFrame,
    booster: xgb.Booster,
    feature_columns: Sequence[str],
) -> float | None:
    if not feature_columns:
        return None
    missing = [col for col in feature_columns if col not in features_df.columns]
    if missing:
        return None
    dmatrix = xgb.DMatrix(features_df[feature_columns].fillna(0))
    scores = booster.predict(dmatrix)
    return float(scores[-1]) if len(scores) else None
