# Change: Dynamic Expert Selection Reasoning

## Why
External news sources are unreliable; we want an LLM-driven expert panel to reason directly on market data and the StatisticsAgent signal.

## What Changes
- Remove GNews dependency and news-text flows from the multi-agent decision path.
- Introduce a Dynamic Expert Selection System in ReasoningAgent with scoring + expert deep dive.
- Update Orchestrator to pass market data directly into ReasoningAgent and persist the expert role output.

## Impact
- Affected specs: agents
- Affected code: agents/reasoning_agent.py, agents/prompts.py, agents/orchestrator.py, app.py, agents/subjectivity_agent.py, README.md
