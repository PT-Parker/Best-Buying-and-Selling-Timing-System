# Change: Manual Gemini Triggers for Multi-Agent and Explanations

## Why
Gemini API quota is limited; automatic calls during reruns quickly exhaust the quota. Manual triggers reduce unintended requests.

## What Changes
- Require an explicit button click to run multi-agent Gemini reasoning.
- Require explicit buttons to generate Gemini explanations for the decision card and forecast.
- Cache the generated explanations for the session to avoid repeated calls.

## Impact
- Affected specs: llm-controls
- Affected code: app.py
