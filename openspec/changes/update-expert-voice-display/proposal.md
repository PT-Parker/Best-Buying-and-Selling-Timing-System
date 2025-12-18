# Change: Expert Persona Display with Winner Voice

## Why
Users want the bull/bear/neutral roles shown as three experts, with the top-scoring expert speaking the winning rationale.

## What Changes
- Render the three expert personas (Chinese names) with their scores in the multi-agent summary.
- Present the top-scoring expert's rationale as a first-person "speech".
- Adjust the scoring prompt to produce a winner rationale suitable for direct display.

## Impact
- Affected specs: agents
- Affected code: app.py, agents/prompts.py, agents/reasoning_agent.py
