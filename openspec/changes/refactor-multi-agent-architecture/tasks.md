## 1. Implementation
- [x] 1.1 Add agents package (StatisticsAgent, RiskAgent, SubjectivityAgent, ReasoningAgent, ReflectionAgent, Orchestrator) with clear interfaces.
- [x] 1.2 Add sqlite memory_db service storing trade history and feeding guidelines to ReflectionAgent.
- [x] 1.3 Refactor core/backtest.py with BacktestTool returning JSON metrics (Sharpe, max drawdown) and detach from LLM prompts.
- [x] 1.4 Wire app.py to load orchestrator/agents entry points without breaking existing UI paths.
- [x] 1.5 Add minimal docs/tests for new agents and backtest tool behavior.
