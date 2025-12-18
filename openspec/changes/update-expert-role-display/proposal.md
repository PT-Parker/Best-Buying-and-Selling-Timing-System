# Change: Chinese Expert Role Display and Score Rationale

## Why
Users need the expert role label in Chinese and the rationale for the highest score to be visible in the UI summary.

## What Changes
- Capture the scoring rationale from expert scoring and include it in the decision payload.
- Display the active expert role in Chinese and show the highest-score rationale in the multi-agent summary.
- Refine the scoring prompt to explicitly explain why the top-scoring role won.

## Impact
- Affected specs: agents
- Affected code: agents/reasoning_agent.py, agents/prompts.py, app.py
