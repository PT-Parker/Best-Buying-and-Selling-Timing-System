# Change: Gemini-powered Expert Reasoning

## Why
External news APIs are unstable; we need deterministic, pure-technical expert reasoning driven by Gemini 2.5 Flash-Lite with JSON-only outputs.

## What Changes
- Add Gemini client wrapper with JSON-only responses via response_mime_type.
- Replace ReasoningAgent with two-phase expert scoring/decision driven by Gemini and market data only.
- Remove GNews/news inputs from orchestrator and UI; persist active_role and scores in memory DB.

## Impact
- Affected specs: agents
- Affected code: utils/llm_client.py, agents/reasoning_agent.py, agents/prompts.py, agents/orchestrator.py, app.py, services/memory_db.py, tests
