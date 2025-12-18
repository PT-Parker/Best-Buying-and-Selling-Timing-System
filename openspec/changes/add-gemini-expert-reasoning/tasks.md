## 1. Implementation
- [x] 1.1 Add utils/llm_client.py with GeminiClient (model gemini-2.5-flash-lite) and JSON-mode chat.
- [x] 1.2 Update prompts for expert scoring + expert decision JSON outputs (Gemini optimized).
- [x] 1.3 Rewrite ReasoningAgent to call Gemini in scoring/decision phases; remove SubjectivityAgent input.
- [x] 1.4 Update Orchestrator/app to pass market data only and persist active_role/scores.
- [x] 1.5 Update tests and requirements for google-generativeai.
