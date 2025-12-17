from __future__ import annotations

from dataclasses import dataclass

from services import memory_db


@dataclass
class ReflectionAgent:
    db: memory_db.MemoryDB
    guidelines: str = ""

    def reflect_on_last_trade(self) -> str:
        row = self.db.fetch_last_trade()
        if row is None:
            self.guidelines = ""
            return self.guidelines

        pred = row.get("model_prediction")
        actual = row.get("actual_outcome")
        if pred == "up" and actual == "down":
            self.guidelines = "RSI 高檔鈍化時模型偏樂觀，下次降低多頭信號權重。"
        elif pred == "down" and actual == "up":
            self.guidelines = "忽略了反轉訊號，下次在低檔注意成交量與趨勢。"
        else:
            self.guidelines = "維持現有決策流程，無明顯偏誤。"
        return self.guidelines
