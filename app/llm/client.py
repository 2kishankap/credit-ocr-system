"""
Thin HTTP client for a local Ollama server.

Ollama exposes an OpenAI-ish REST API on port 11434. We use /api/generate
with `format: "json"`, which makes Ollama constrain sampling to valid
JSON -- this is what makes structured extraction reliable enough to
parse without a fragile regex-based cleanup step.
"""
from __future__ import annotations

import json
import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=15))
    def generate_json(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> dict:
        """
        Call the local model and force JSON-formatted output.
        Retries with exponential backoff -- the model server can be briefly
        unavailable right after container startup while weights load.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=settings.ollama_request_timeout) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        body = response.json()
        raw_text = body.get("response", "")
        return self._parse_json(raw_text)

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        """Ollama's `format: json` mode should return clean JSON, but we defensively
        trim anything outside the outermost braces in case the model adds prose."""
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start, end = raw_text.find("{"), raw_text.rfind("}")
            if start == -1 or end == -1:
                logger.error("LLM response was not valid JSON: %.500s", raw_text)
                raise
            return json.loads(raw_text[start : end + 1])

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


_client_singleton: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OllamaClient()
    return _client_singleton
