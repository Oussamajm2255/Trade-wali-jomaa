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

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from trading_agent.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Failure taxonomy (spec §37): every failure is classified so the caller
# can record it honestly and apply the fallback-isolation policy.
FAILURE_TIMEOUT = "TIMEOUT"
FAILURE_INVALID_JSON = "INVALID_JSON"
FAILURE_API_ERROR = "API_ERROR"
FAILURE_RATE_LIMIT = "RATE_LIMIT"
FAILURE_EMPTY_RESPONSE = "EMPTY_RESPONSE"
FAILURE_UNKNOWN = "UNKNOWN"


class AgentError(RuntimeError):
    """Terminal failure of an LLM call after all retries."""


class AgentFailure(AgentError):
    """A classified LLM failure; reason is one of FAILURE_* above."""

    def __init__(self, reason: str, cause: Exception | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cause = cause


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
            raise AgentFailure(FAILURE_API_ERROR, None)
        last_failure: AgentFailure | None = None
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
                if not text.strip():
                    raise AgentFailure(FAILURE_EMPTY_RESPONSE, None)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise AgentFailure(FAILURE_INVALID_JSON, exc)
            except AgentFailure as exc:
                last_failure = exc
                logger.warning(
                    "LLM attempt %s/%s failed (%s): %s",
                    attempt + 1,
                    self.settings.llm_max_retries,
                    exc.reason,
                    exc.cause or exc,
                )
            except openai.APITimeoutError as exc:
                last_failure = AgentFailure(FAILURE_TIMEOUT, exc)
                logger.warning(
                    "LLM attempt %s/%s timed out", attempt + 1, self.settings.llm_max_retries
                )
            except openai.RateLimitError as exc:
                last_failure = AgentFailure(FAILURE_RATE_LIMIT, exc)
                logger.warning(
                    "LLM attempt %s/%s rate limited", attempt + 1, self.settings.llm_max_retries
                )
            except openai.APIError as exc:
                last_failure = AgentFailure(FAILURE_API_ERROR, exc)
                logger.warning(
                    "LLM attempt %s/%s API error: %s",
                    attempt + 1,
                    self.settings.llm_max_retries,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001 - network/parse errors
                last_failure = AgentFailure(FAILURE_UNKNOWN, exc)
                logger.warning(
                    "LLM attempt %s/%s failed: %s",
                    attempt + 1,
                    self.settings.llm_max_retries,
                    exc,
                )
            time.sleep(min(2**attempt, 8))
        reason = last_failure.reason if last_failure else FAILURE_UNKNOWN
        raise AgentFailure(
            reason,
            RuntimeError(
                f"LLM call failed after {self.settings.llm_max_retries} attempts "
                f"({reason}): {last_failure.cause if last_failure else ''}"
            ),
        )

    def complete_model(self, system: str, user: str, schema: type[T]) -> tuple[T | None, str | None]:
        """Return (validated model | None, failure_reason | None). Never raises.

        failure_reason is None on success and one of FAILURE_* when the
        call or the validation failed — the caller decides the fallback
        policy (spec §37).
        """
        if self._client is None:
            return None, FAILURE_API_ERROR
        try:
            data = self.complete_json(system, user)
        except AgentFailure as exc:
            logger.warning("LLM call failed (%s): %s", exc.reason, exc.cause or exc)
            return None, exc.reason
        try:
            return schema.model_validate(data), None
        except ValidationError as exc:
            logger.warning("LLM output invalid (%s); using fallback", exc)
            return None, FAILURE_INVALID_JSON

    def balance_usd(self) -> float | None:
        """Current DeepSeek account balance in USD, or None if unreachable.

        Used for monitoring: an empty balance makes every analysis degrade
        to heuristics silently — better to alert before that happens.
        """
        import httpx

        url = f"{self.settings.deepseek_base_url.rstrip('/')}/user/balance"
        try:
            resp = httpx.get(
                url,
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key.get_secret_value()}"
                },
                timeout=10,
            )
            data = resp.json()
            if resp.status_code != 200 or not data.get("is_available"):
                return None
            usd = [b for b in data.get("balance_infos", []) if b.get("currency") == "USD"]
            if not usd:
                return None
            return float(usd[0]["total_balance"])
        except Exception as exc:  # noqa: BLE001 - monitoring must never raise
            logger.warning("DeepSeek balance check failed: %s", exc)
            return None
