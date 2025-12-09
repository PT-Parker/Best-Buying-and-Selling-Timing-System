import pandas as pd

from core import labeling


def test_assign_labels_marks_positive_and_negative():
    data = pd.DataFrame(
        {
            "symbol": ["TEST"] * 6,
            "date": pd.date_range("2024-01-01", periods=6, freq="B"),
            "close": [100, 103, 106, 90, 85, 82],
        }
    )
    config = labeling.LabelConfig(horizon_days=2, take_profit_pct=0.02, stop_loss_pct=-0.02)
    labelled = labeling.assign_labels(data, config)
    assert "future_return" in labelled.columns
    assert "label" in labelled.columns
    assert set(labelled["label"].unique()) <= {1, 0, -1}
