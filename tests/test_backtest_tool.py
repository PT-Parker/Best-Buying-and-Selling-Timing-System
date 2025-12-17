import pandas as pd

from core.backtest import BacktestTool


def test_backtest_tool_outputs_json():
    prices = pd.Series([100, 102, 101, 103], name="close")
    signals = pd.Series([0, 1, 1, 0], name="signal")
    tool = BacktestTool()
    result = tool.run(signals, prices)
    assert "sharpe" in result and "max_drawdown" in result and "equity_curve" in result
    assert isinstance(result["equity_curve"], list)
