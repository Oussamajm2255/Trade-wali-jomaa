"""LLM failure taxonomy (spec §37): every failure gets a classified
reason and the heuristic fallback is clearly labelled DEGRADED_MODE."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest

from trading_agent.agents.base import (
    FAILURE_EMPTY_RESPONSE,
    FAILURE_INVALID_JSON,
    FAILURE_TIMEOUT,
    LLMClient,
)
from trading_agent.agents.dxy import DxyContextAgent
from trading_agent.agents.technical import TechnicalAgent
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, TechnicalOutput


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://test")


def _client() -> LLMClient:
    settings = Settings(deepseek_api_key="sk-test-key-123", llm_max_retries=1, llm_timeout_seconds=1)
    return LLMClient(settings)


class _StubCompletions:
    def __init__(self, fn) -> None:
        self._fn = fn

    def create(self, **kwargs):
        return self._fn()


class _StubChat:
    def __init__(self, fn) -> None:
        self.completions = _StubCompletions(fn)


class _Message:
    def __init__(self, content: str | None) -> None:
        self.content = content


def _respond(content: str | None):
    def _fn():
        return SimpleNamespace(choices=[SimpleNamespace(message=_Message(content))])

    return _fn


def _raise(exc: Exception):
    def _fn():
        raise exc

    return _fn


@pytest.mark.parametrize(
    "stub, expected",
    [
        (_raise(openai.APITimeoutError(request=_request())), FAILURE_TIMEOUT),
        (
            _raise(openai.APIConnectionError(request=_request())),
            "API_ERROR",
        ),
        (
            _raise(
                openai.RateLimitError(
                    "limited",
                    response=httpx.Response(429, request=_request()),
                    body=None,
                )
            ),
            "RATE_LIMIT",
        ),
        (_respond(""), FAILURE_EMPTY_RESPONSE),
        (_respond("not json at all"), FAILURE_INVALID_JSON),
    ],
)
def test_complete_model_classifies_failures(stub, expected) -> None:
    client = _client()
    client._client = SimpleNamespace(chat=_StubChat(stub))  # type: ignore[assignment]
    model, reason = client.complete_model("sys", "user", TechnicalOutput)
    assert model is None
    assert reason == expected


def test_complete_model_rejects_schema_mismatch() -> None:
    client = _client()
    client._client = SimpleNamespace(  # type: ignore[assignment]
        chat=_StubChat(_respond('{"bias": "moon", "conviction": 0.5}'))
    )
    model, reason = client.complete_model("sys", "user", TechnicalOutput)
    assert model is None
    assert reason == FAILURE_INVALID_JSON


class _StubClient:
    """Duck-typed LLMClient: enabled flag + complete_model behaviour."""

    def __init__(self, enabled: bool, result=None, failure: str | None = None) -> None:
        self.enabled = enabled
        self._result = result
        self._failure = failure

    def complete_model(self, system: str, user: str, schema):
        return self._result, self._failure


def test_agent_labels_failure_on_llm_error() -> None:
    settings = Settings(deepseek_api_key="sk-test-key-123")
    stub = _StubClient(enabled=True, result=None, failure="TIMEOUT")
    verdict = TechnicalAgent(stub, settings).analyze({})  # type: ignore[arg-type]
    assert isinstance(verdict, AgentVerdict)
    assert verdict.source == "fallback"
    assert verdict.failure_reason == "TIMEOUT"
    assert verdict.model == "heuristic-fallback"


def test_agent_llm_disabled_is_not_a_failure() -> None:
    # LLM-disabled-by-config uses the fallback but must NOT count as a
    # failure for the isolation policy (spec §37).
    settings = Settings()
    stub = _StubClient(enabled=False)
    verdict = TechnicalAgent(stub, settings).analyze({})  # type: ignore[arg-type]
    assert verdict.source == "fallback"
    assert verdict.failure_reason is None


def test_dxy_agent_fallback_shape() -> None:
    settings = Settings()
    stub = _StubClient(enabled=False)
    verdict = DxyContextAgent(stub, settings).analyze(  # type: ignore[arg-type]
        {"dxy_context": {"score": 70, "classification": "Bullish (USD weak)"}}, None
    )
    payload = verdict.payload
    assert payload["gold_bias"] == "long"
    assert payload["score"] > 0
    assert payload["dxy_state"] == "Bullish (USD weak)"
    assert verdict.failure_reason is None


def test_technical_agent_accepts_valid_llm_output() -> None:
    settings = Settings(deepseek_api_key="sk-test-key-123")
    valid = TechnicalOutput(
        bias="short",
        conviction=0.6,
        setup_type="choch",
        structure_alignment=-0.4,
        reasoning="structure broke down",
        invalidating_condition="reclaim of the broken level",
    )
    stub = _StubClient(enabled=True, result=valid)
    verdict = TechnicalAgent(stub, settings).analyze({})  # type: ignore[arg-type]
    assert verdict.source == "llm"
    assert verdict.payload["bias"] == "short"
    assert verdict.payload["setup_type"] == "choch"
    assert verdict.failure_reason is None
