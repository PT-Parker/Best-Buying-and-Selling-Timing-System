from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class RiskAgent:
    """Performs guard-rail checks before allowing trades."""

    rsi_cap: float = 80.0
    atr_window: int = 14
    vol_cap: float = 0.05  # ATR as % of price

    def _atr(self, prices: pd.DataFrame) -> float | None:
        if not {"high", "low", "close"}.issubset(prices.columns):
            return None
        high, low, close = prices["high"], prices["low"], prices["close"]
        prev_close = close.shift()
        tr = np.maximum(
            high - low,
            np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
        )
        atr = tr.rolling(self.atr_window).mean()
        return float(atr.iloc[-1]) if len(atr) >= self.atr_window else None

    def approve_trade(self, signal: dict, prices: pd.DataFrame) -> Tuple[bool, str]:
        if prices.empty:
            return False, "缺少價格資料"
        latest_rsi = prices["rsi_14"].iloc[-1] if "rsi_14" in prices else None
        if latest_rsi is not None and latest_rsi > self.rsi_cap:
            return False, f"RSI {latest_rsi:.1f} 高於上限 {self.rsi_cap}"

        atr_val = self._atr(prices)
        if atr_val is not None:
            last_px = float(prices["close"].iloc[-1])
            if last_px > 0 and atr_val / last_px > self.vol_cap:
                return False, f"ATR {atr_val:.4f} / 價格過高"

        return True, "OK"
