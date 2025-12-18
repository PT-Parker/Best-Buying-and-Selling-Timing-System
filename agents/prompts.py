EXPERT_SCORING_PROMPT = """
你是市場策略總監。請根據以下市場數據與模型分數，評估目前環境更適合哪位專家的策略。
專家角色：
1) Bull Expert (做多專家)：趨勢突破、支撐反彈
2) Bear Expert (做空專家)：壓力位不過、動能背離
3) Neutral Expert (中立觀察者)：波動率風險，偏向觀望/區間

市場數據（JSON）：
{market_data}

模型分數（0-1）：
{model_score}

請只回傳 JSON，不要任何多餘文字：
{{
  "bull_score": int,
  "bear_score": int,
  "neutral_score": int
}}
"""

EXPERT_DECISION_PROMPT = """
你是 {active_role}，你的策略得分最高。請根據市場數據做出最終決策，並反駁其他兩位專家觀點。

市場數據（JSON）：
{market_data}

模型分數（0-1）：
{model_score}

反思指引：
{guidelines}

請只回傳 JSON，不要任何多餘文字：
{{
  "action": "buy" | "sell" | "hold",
  "confidence": float,
  "reasoning": "簡短說明，含反駁其他觀點",
  "active_role": "{active_role}"
}}
"""
