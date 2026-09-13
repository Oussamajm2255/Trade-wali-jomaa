"""DeepSeek balance monitoring: API parsing + degraded-mode detection."""
from __future__ import annotations

import httpx
import pytest

from trading_agent.agents.base import LLMClient
from trading_agent.agents.orchestrator import llm_degraded
from trading_agent.schema.types import AgentVerdict


class FakeResp:
    def __init__(self, payload: dict | None, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def patch_balance_http(monkeypatch, payload=None, status=200, raise_exc: Exception | None = None):
    def fake_get(url, headers=None, timeout=None):
        if raise_exc is not None:
            raise raise_exc
        return FakeResp(payload, status)

    monkeypatch.setattr(httpx, "get", fake_get)


def make_client(base_settings) -> LLMClient:
    return LLMClient(base_settings)


def test_balance_parsed_from_usd_entry(monkeypatch, base_settings):
    patch_balance_http(
        monkeypatch,
        {
            "is_available": True,
            "balance_infos": [
                {"currency": "USD", "total_balance": "12.34", "granted_balance": "0.00",
                 "topped_up_balance": "12.34"},
                {"currency": "CNY", "total_balance": "0.00"},
            ],
        },
    )
    assert make_client(base_settings).balance_usd() == 12.34


def test_balance_none_when_unavailable(monkeypatch, base_settings):
    patch_balance_http(monkeypatch, {"is_available": False, "balance_infos": []})
    assert make_client(base_settings).balance_usd() is None


def test_balance_none_without_usd_currency(monkeypatch, base_settings):
    patch_balance_http(
        monkeypatch,
        {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "5.00"}]},
    )
    assert make_client(base_settings).balance_usd() is None


def test_balance_none_on_http_error(monkeypatch, base_settings):
    patch_balance_http(monkeypatch, payload=None, status=500)
    assert make_client(base_settings).balance_usd() is None


def test_balance_none_on_exception(monkeypatch, base_settings):
    patch_balance_http(monkeypatch, raise_exc=TimeoutError("slow"))
    assert make_client(base_settings).balance_usd() is None


def verdict(source: str) -> AgentVerdict:
    return AgentVerdict(agent="technical", source=source, model="m", payload={})


def test_llm_degraded_all_fallback():
    verdicts = {"technical": verdict("fallback"), "sentiment": verdict("fallback"),
                "regime": verdict("fallback")}
    assert llm_degraded(verdicts) is True


def test_llm_degraded_one_llm_verdict():
    verdicts = {"technical": verdict("llm"), "sentiment": verdict("fallback")}
    assert llm_degraded(verdicts) is False


def test_llm_degraded_empty_verdicts():
    assert llm_degraded({}) is False
