"""Thin, hardened wrapper around the DeepSeek (OpenAI-compatible) API.

Design rules:
- The LLM only ever returns *validated JSON schemas* — free text may be
  shown to humans but is never parsed into decisions.
- Every call is bounded: timeout, retries with backoff, and a hard
  failure after max attempts so the pipeline degrades to deterministic
  fallbacks instead of hanging or fabricating.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from trading_agent.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgentError(RuntimeError):
    """Terminal failure of an LLM call after all retries."""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.llm_enabled
        self._client: OpenAI | None = None
        if self.enabled:
            self._client = OpenAI(
                api_key=settings.deepseek_api_key.get_secret_value(),
                base_url=settings.deepseek_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,  # we retry ourselves, with backoff
            )

    def complete_json(self, system: str, user: str) -> dict:
        """Ask the model for a JSON object; returns the parsed dict."""
        if self._client is None:
            raise AgentError("LLM client is not configured")
        last_error: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.settings.deepseek_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.settings.llm_temperature,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content or ""
                return json.loads(text)
            except Exception as exc:  # noqa: BLE001 - network/API/parse errors
                last_error = exc
                logger.warning(
                    "LLM attempt %s/%s failed: %s",
                    attempt + 1,
                    self.settings.llm_max_retries,
                    exc,
                )
                time.sleep(min(2**attempt, 8))
        raise AgentError(
            f"LLM call failed after {self.settings.llm_max_retries} attempts: {last_error}"
        )

    def complete_model(self, system: str, user: str, schema: type[T]) -> tuple[T | None, bool]:
        """Return (validated model | None, used_fallback). Never raises."""
        try:
            data = self.complete_json(system, user)
            return schema.model_validate(data), False
        except (AgentError, ValidationError, json.JSONDecodeError) as exc:
            logger.warning("LLM output invalid (%s); using fallback", exc)
            return None, True
