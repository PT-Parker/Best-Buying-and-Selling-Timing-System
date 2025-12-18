# Change: Enforce Gemini Mandatory Expert Reasoning

## Why
To avoid silent heuristic fallbacks, the system must require Gemini for expert scoring and display LLM status in the UI.

## What Changes
- Enforce Gemini availability; block multi-agent execution when API key is missing.
- Remove heuristic fallbacks in ReasoningAgent; raise errors on invalid LLM JSON.
- Enhance expert scoring prompt with implied sentiment from price/volume microstructure.
- Add retry-on-JSON-failure in Gemini client.

## Impact
- Affected specs: agents
- Affected code: app.py, utils/llm_client.py, agents/reasoning_agent.py, agents/prompts.py, services/memory_db.py
