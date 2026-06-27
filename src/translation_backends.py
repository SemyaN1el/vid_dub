"""Backend data containers shared by translation and SmartSync code."""

from dataclasses import dataclass
from typing import Any


@dataclass
class GeminiTranslationBackend:
    model_name: str
    api_key: str
    temperature: float
    max_output_tokens: int
    thinking_budget: int
    timeout_sec: int
    min_interval_sec: float
    max_retries: int
    backend: str = "gemini"
    last_request_ts: float = 0.0


@dataclass
class OpenAICompatibleTranslationBackend:
    model_name: str
    api_key: str
    base_url: str
    temperature: float
    max_output_tokens: int
    timeout_sec: int
    min_interval_sec: float
    max_retries: int
    backend: str = "openai_compatible"
    last_request_ts: float = 0.0
    client: Any | None = None


SmartSyncBackend = GeminiTranslationBackend | OpenAICompatibleTranslationBackend
