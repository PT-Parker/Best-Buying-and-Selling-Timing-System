from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Indicator parameters and trade management rules."""

    ema_fast: int = 5
    ema_slow: int = 20
    rsi_period: int = 14
    rsi_buy_lt: float = 30.0
    rsi_sell_gt: float = 70.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    take_profit_pct: float = 0.03
    stop_loss_pct: float = 0.015
    cooldown_days: int = 0


@dataclass
class FeesConfig:
    commission: float = 0.001425
    slippage: float = 0.0005
    initial_capital: float = 1_000_000.0


@dataclass
class BacktestResult:
    symbol: str
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    signals: pd.DataFrame


def _ensure_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjclose": "Close",
        "Adj Close": "Close",
        "volume": "Volume",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)
    df["Date"] = pd.to_datetime(df.get("Date") or df.index)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Price frame missing columns: {missing}")
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def _bollinger(series: pd.Series, period: int, std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    return mid, upper, lower


def generate_indicators(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    df = _ensure_price_frame(prices)
    out = df.copy()
    out["EMA_FAST"] = _ema(out["Close"], config.ema_fast)
    out["EMA_SLOW"] = _ema(out["Close"], config.ema_slow)
    out["RSI"] = _rsi(out["Close"], config.rsi_period)
    mid, up, lo = _bollinger(out["Close"], config.bollinger_period, config.bollinger_std)
    out["BB_MID"], out["BB_UP"], out["BB_LO"] = mid, up, lo
    out["cross_up_lo"] = (out["Close"].shift(1) < out["BB_LO"].shift(1)) & (out["Close"] >= out["BB_LO"])
    out["cross_dn_up"] = (out["Close"].shift(1) > out["BB_UP"].shift(1)) & (out["Close"] <= out["BB_UP"])
    return out


def _reason_text(signal: str, row: pd.Series) -> str:
    if signal == "BUY":
        return "RSI<{:.0f} & price↗下軌".format(row.get("RSI", 0))
    if signal == "SELL":
        return "RSI>{:.0f} & price↘上軌".format(row.get("RSI", 0))
    return "無觸發"


def generate_signals(prices: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    df = generate_indicators(prices, config)
    cond_buy = (df["RSI"] < config.rsi_buy_lt) & df["cross_up_lo"]
    cond_sell = (df["RSI"] > config.rsi_sell_gt) & df["cross_dn_up"]
    df["signal"] = "HOLD"
    df.loc[cond_buy, "signal"] = "BUY"
    df.loc[cond_sell, "signal"] = "SELL"

    confidence: List[str] = ["Weak"] * len(df)
    for idx in range(len(df)):
        sig = df["signal"].iat[idx]
        if sig == "BUY":
            confidence[idx] = "Strong" if df["EMA_FAST"].iat[idx] > df["EMA_SLOW"].iat[idx] else "Normal"
        elif sig == "SELL":
            confidence[idx] = "Strong" if df["EMA_FAST"].iat[idx] < df["EMA_SLOW"].iat[idx] else "Normal"
    df["confidence"] = confidence
    df["reason"] = [
        _reason_text(sig, row) if sig != "HOLD" else "無觸發"
        for sig, (_, row) in zip(df["signal"], df.iterrows())
    ]
    return df


def run_backtest(
    prices: pd.DataFrame,
    config: StrategyConfig,
    fees: FeesConfig | None = None,
    symbol: str | None = None,
) -> BacktestResult:
    fees = fees or FeesConfig()
    signal_df = generate_signals(prices, config)
    signal_df = signal_df.dropna(subset=["Close"]).copy()

    cash = fees.initial_capital
    shares = 0
    entry_price = 0.0
    entry_date: Optional[pd.Timestamp] = None
    trades: List[Dict] = []
    equity_values: List[float] = []

    for idx, row in signal_df.iterrows():
        price = float(row["Close"])
        signal = row["signal"]
        if shares == 0 and signal == "BUY":
            buy_px = price * (1 + fees.slippage)
            qty = int((cash * (1 - fees.commission)) // buy_px)
            if qty <= 0:
                equity_values.append(cash)
                continue
            cost = qty * buy_px
            fee = cost * fees.commission
            cash -= cost + fee
            shares = qty
            entry_price = buy_px
            entry_date = idx
        elif shares > 0 and signal == "SELL":
            sell_px = price * (1 - fees.slippage)
            proceeds = shares * sell_px
            fee = proceeds * fees.commission
            cash += proceeds - fee
            holding_days = (idx - entry_date).days if entry_date is not None else None
            pnl = (sell_px - entry_price) * shares - fee
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": idx,
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(sell_px, 4),
                    "shares": shares,
                    "holding_days": holding_days,
                    "pnl": round(pnl, 2),
                }
            )
            entry_price = 0.0
            shares = 0
            entry_date = None
        equity = cash + (shares * price if shares > 0 else 0.0)
        equity_values.append(equity)

    signal_df["equity"] = equity_values
    equity_curve = signal_df[["equity"]].copy()

    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(signal_df, trades_df, fees)
    return BacktestResult(
        symbol=symbol or signal_df.attrs.get("symbol", "UNKNOWN"),
        metrics=metrics,
        trades=trades_df,
        equity_curve=equity_curve,
        signals=signal_df.reset_index().rename(columns={"index": "Date"}),
    )


def _compute_metrics(timeseries: pd.DataFrame, trades: pd.DataFrame, fees: FeesConfig) -> Dict[str, float]:
    equity = timeseries["equity"]
    if equity.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "avg_holding_days": 0.0,
            "signal_count": 0,
            "win_rate": 0.0,
        }
    total_return = equity.iloc[-1] / fees.initial_capital - 1
    periods = len(equity)
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / periods) - 1 if periods > 0 else 0.0
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_drawdown = float(drawdown.min())
    avg_holding = float(trades["holding_days"].mean()) if not trades.empty else 0.0
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": max_drawdown,
        "trade_count": int(len(trades)),
        "avg_holding_days": avg_holding,
        "signal_count": int(timeseries["signal"].isin(["BUY", "SELL"]).sum()),
        "win_rate": win_rate,
    }


class BacktestTool:
    """Isolated backtest utility that returns strict JSON metrics for LLM consumers."""

    def run(self, signals: pd.Series, prices: pd.Series) -> Dict[str, object]:
        if signals is None or prices is None:
            raise ValueError("signals and prices are required")
        if len(signals) != len(prices):
            raise ValueError("signals and prices must be aligned")

        # Signals assumed numeric: 1 for long, 0 for flat
        returns = prices.pct_change().fillna(0)
        strat_ret = signals.shift().fillna(0) * returns
        equity_curve = (1 + strat_ret).cumprod()

        max_dd = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax().clip(lower=1e-9)).max()
        sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-9)

        return {
            "equity_curve": equity_curve.tolist(),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }
