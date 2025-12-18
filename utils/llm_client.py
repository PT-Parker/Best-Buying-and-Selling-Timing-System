from __future__ import annotations

import json
import os
from typing import Optional

try:  # pragma: no cover - optional for non-Streamlit contexts
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None

import google.generativeai as genai


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite"):
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key and st is not None:
            key = st.secrets.get("GOOGLE_API_KEY")  # type: ignore[attr-defined]
        if not key:
            raise ValueError("Google API Key not found.")

        genai.configure(api_key=key)
        self.model_name = model_name

    def chat(self, prompt: str, json_mode: bool = True, max_retries: int = 1) -> str:
        generation_config = {}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        model = genai.GenerativeModel(model_name=self.model_name, generation_config=generation_config)
        attempts = 0
        last_error: Exception | None = None
        while attempts <= max_retries:
            try:
                response = model.generate_content(prompt)
                text = response.text
                if json_mode:
                    json.loads(text)
                return text
            except Exception as exc:  # pragma: no cover - network error path
                last_error = exc
                attempts += 1
        raise RuntimeError(f"Gemini API error: {last_error}")
