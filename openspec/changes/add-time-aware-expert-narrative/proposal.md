# Change: Time-Aware Expert Narrative and Confidence Calibration

## Why
LLM expert reasoning needs temporal context to avoid time-agnostic decisions and should produce confidence scores that vary with signal alignment.

## What Changes
- Add a time-aware narrative summary of recent price/volume behavior to expert scoring and decision prompts.
- Update reasoning/orchestration to pass the narrative alongside technical indicators.
- Adjust expert decision prompt to calibrate confidence based on signal agreement or conflict.

## Impact
- Affected specs: agents
- Affected code: agents/prompts.py, agents/reasoning_agent.py, agents/orchestrator.py, tests if needed
