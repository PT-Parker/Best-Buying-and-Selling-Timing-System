# Change: Refactor to multi-agent architecture with tool-isolated backtesting and reflection

## Why
Current Streamlit app is monolithic, mixing XGBoost inference, risk checks, UI, and backtesting; this causes coupling, hallucination risk, and no persistence for reflections.

## What Changes
- Introduce modular agents (statistics, risk, subjectivity, reasoning, reflection) and an orchestrator callable from the UI.
- Add sqlite-backed memory for trade history and reflection feedback into future decisions.
- Isolate backtest computation in a BacktestTool to prevent LLM-driven hallucinations; return strict JSON metrics.

## Impact
- Affected specs: agents
- Affected code: app.py, agents/*, services/memory_db.py, core/backtest.py
