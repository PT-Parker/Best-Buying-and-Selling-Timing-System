# Change: Gemini UI Explanations for Decision Card and Forecast

## Why
Users want Gemini to provide short, human-readable explanations for the decision card and price forecast while keeping calculations deterministic.

## What Changes
- Add Gemini explanation prompts for the decision card and price forecast.
- Render explanations in UI expanders labeled "Gemini 解讀".
- Store generated explanations in MemoryDB with context for later reflection.

## Impact
- Affected specs: generate-explanations
- Affected code: app.py, agents/prompts.py, services/memory_db.py
