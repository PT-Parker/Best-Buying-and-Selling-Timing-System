EXPERT_SCORING_PROMPT = """
你是一個高頻交易演算法的「狀態路由器 (Regime Router)」。
你的任務是根據純技術數據，將當前市場狀態分類為適合以下哪位專家操作：

1. 🐂 做多專家 (Bull): 強勢上漲趨勢、超賣反彈、突破關鍵壓力。
2. 🐻 做空專家 (Bear): 下跌趨勢、技術指標背離、跌破關鍵支撐。
3. ⚖️ 中立專家 (Neutral): 無方向震盪、波動率極低、多空訊號衝突。

即便沒有新聞，請從「價格行為 (Price Action)」推斷市場情緒：
- 劇烈長紅/長黑 K 線 -> 隱含強烈情緒 (高 Bull/Bear 分數)。
- 窄幅盤整 -> 隱含觀望情緒 (高 Neutral 分數)。
雖然沒有外部新聞，請透過「市場微結構」推斷隱含情緒：
1) 成交量 (Volume)：價漲量增 => Bullish；價跌量增 => Bearish。
2) 波動率 (Volatility)：窄幅盤整後長黑/長紅 => 強烈情緒傾向。
3) 價格型態：識別假突破、吸籌或派發等模式。

時間敘述 (最近幾日):
{time_summary}

技術數據:
{market_data}

請輸出 JSON（只回傳 JSON）:
{{
  "bull_score": 0-100,
  "bear_score": 0-100,
  "neutral_score": 0-100,
  "reasoning": "以第一人稱說明最高分角色為何勝出，例如：『我看到...所以得分最高』"
}}
"""

EXPERT_DECISION_PROMPT = """
你是 {active_role} 專家，你的策略得分最高。請根據技術數據與時間敘述做出最終交易信號並反駁其他觀點。

時間敘述:
{time_summary}

技術數據:
{market_data}

模型分數（0-1）:
{model_score}

反思指引:
{guidelines}

信心校準規則:
- 指標與時間敘述高度一致、共振明確時，confidence 才可 > 0.8。
- 訊號衝突或背離明顯時，confidence 應 < 0.6。
- 其餘情況落在 0.6-0.8 之間。

請輸出 JSON（只回傳 JSON）:
{{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": float,
  "reasoning": "簡短說明，包含反駁其他觀點",
  "active_role": "{active_role}"
}}
"""

DECISION_EXPLANATION_PROMPT = """
你是交易助理，請根據提供的數據，用中文產生 2-3 行解釋。
重點：只使用提供的數據，不要臆測新聞或事件，不要重新計算。

輸入資料:
- 標的: {symbol}
- 日期: {as_of}
- 收盤價: {close}
- 模型分數: {model_score}
- 期望報酬: {expected_return}
- 技術訊號: {signal}
- 建議動作: {action}
- 策略原因: {reason}
- 停利價: {take_profit}
- 停損價: {stop_loss}
- 預估持有天數: {horizon_days}

請輸出 JSON（只回傳 JSON）:
{{
  "explanation": "請用 2-3 行說明，每行用換行分隔"
}}
"""

FORECAST_EXPLANATION_PROMPT = """
你是交易助理，請根據提供的預測資料，用中文產生 2-3 行解釋。
重點：只使用提供的數據，不要臆測新聞或事件。

輸入資料:
- 標的: {symbol}
- 最新收盤價: {last_price}
- 預測天數: {forecast_days}
- 預測終點價格: {forecast_price}
- 模型分數: {model_score}
- 期望日報酬估計: {expected_return}

請輸出 JSON（只回傳 JSON）:
{{
  "explanation": "請用 2-3 行說明，每行用換行分隔"
}}
"""
