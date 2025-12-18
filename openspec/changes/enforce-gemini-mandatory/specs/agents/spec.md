## MODIFIED Requirements
### Requirement: Gemini Expert Reasoning
The system SHALL require Gemini 2.5 Flash-Lite for expert scoring and decisions, and SHALL block multi-agent execution if LLM is unavailable.

#### Scenario: LLM missing blocks execution
- **WHEN** the user enables multi-agent decisioning without a valid API key,
- **THEN** the UI shows an offline status and halts expert evaluation.

### Requirement: Expert Scoring Based on Implied Sentiment
The system SHALL instruct Gemini to infer implied sentiment from price action, volume, and volatility rather than external news.

#### Scenario: Microstructure-driven scoring
- **WHEN** market data includes volume and volatility features,
- **THEN** the expert scoring prompt directs Gemini to use implied sentiment cues.
