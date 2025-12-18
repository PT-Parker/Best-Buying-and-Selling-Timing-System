# Change: Sidebar Gemini API Key Entry

## Why
Allow users to provide GEMINI_API_KEY at runtime via the Streamlit sidebar without editing environment files.

## What Changes
- Add a sidebar password field for GEMINI_API_KEY and a button to activate it.
- Store the key only in session state and use it when creating the Gemini client.
- Update LLM status to reflect the presence of a session key.

## Impact
- Affected specs: llm-config
- Affected code: app.py, utils/llm_client.py
