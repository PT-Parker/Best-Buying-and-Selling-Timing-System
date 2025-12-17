## ADDED Requirements
### Requirement: Multi-Agent Trading Orchestration
The system SHALL separate trading logic into modular agents (statistics, risk, subjectivity, reasoning, reflection) coordinated by an orchestrator callable from the UI.

#### Scenario: Orchestrator runs decision flow
- **WHEN** the orchestrator is invoked with symbol data and optional news text,
- **THEN** it queries StatisticsAgent for model signals,
- **AND** consults RiskAgent for approval,
- **AND** combines StatisticsAgent and SubjectivityAgent inputs within ReasoningAgent using market regime weighting,
- **AND** returns a final decision payload for the UI.

### Requirement: Risk Agent Gatekeeping
The system SHALL enforce risk checks (e.g., RSI cap, ATR/volatility cap) that can veto trades regardless of model optimism.

#### Scenario: Overheated market blocked
- **WHEN** the latest RSI exceeds a configured threshold or volatility exceeds a configured cap,
- **THEN** RiskAgent.approve_trade returns False with a reason,
- **AND** the orchestrator marks the action as blocked.

### Requirement: Fact-Subjectivity Split
The system SHALL incorporate subjectivity sentiment into decisions with regime-based weighting, increasing sentiment weight in bull markets and statistics weight in bear markets.

#### Scenario: Bull market sentiment upweighting
- **WHEN** SMA20 > SMA60 is detected,
- **THEN** ReasoningAgent increases the weight of SubjectivityAgent sentiment relative to StatisticsAgent output before deciding.

### Requirement: Reflection with Persistent Memory
The system SHALL persist trade history to sqlite and let ReflectionAgent inject guidelines from past mispredictions into the next decision prompt.

#### Scenario: Reflection after incorrect bullish call
- **WHEN** the last trade recorded a bullish model_prediction but the actual_outcome was bearish,
- **THEN** ReflectionAgent generates a guideline noting the mismatch,
- **AND** the orchestrator includes this guideline in the subsequent decision context.

### Requirement: Tool-Isolated Backtesting
The system SHALL encapsulate backtest computation in BacktestTool that returns strict JSON metrics (Sharpe, max drawdown), with LLM agents only consuming the JSON and never performing calculations.

#### Scenario: Backtest JSON output
- **WHEN** BacktestTool.run is executed with signal and price series,
- **THEN** it returns JSON containing at least sharpe and max_drawdown plus an equity curve array,
- **AND** the UI renders metrics using this JSON without asking LLMs to compute them.
