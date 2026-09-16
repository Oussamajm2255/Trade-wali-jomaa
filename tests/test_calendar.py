"""Economic calendar (spec §43): normalized events, provider fail-open,
and the opt-in NEWS_RISK gate in the risk engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.config import Settings
from trading_agent.data.calendar import (
    NullCalendarProvider,
    FinnhubCalendarProvider,
    blocking_events,
    build_calendar_provider,
    normalize_events,
)
from trading_agent.data.snapshot import build_market_snapshot
from trading_agent.fusion.types import NoTradeReason
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, Side


def test_null_provider_never_fabricates_events():
    assert NullCalendarProvider().upcoming_events(datetime.now(timezone.utc), 60) == []


def test_build_calendar_provider_unknown_name_falls_back_to_null():
    assert isinstance(build_calendar_provider(Settings(news_provider="nope")), NullCalendarProvider)


def test_build_calendar_provider_finnhub():
    provider = build_calendar_provider(Settings(news_provider="finnhub", finnhub_api_key="k"))
    assert isinstance(provider, FinnhubCalendarProvider)


def test_normalize_events_shape_and_usd_filter():
    now = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
    raw = [
        {"country": "US", "event": "CPI", "impact": "high",
         "date": "2026-09-16", "time": "10:30:00"},
        {"country": "US", "event": "PMI", "impact": "medium",
         "date": "2026-09-16", "time": "11:00:00"},
        {"country": "DE", "event": "GDP", "impact": "high",  # not USD
         "date": "2026-09-16", "time": "11:00:00"},
        {"country": "US", "event": "weird", "impact": "unknown",  # no rank
         "date": "2026-09-16", "time": "11:00:00"},
    ]
    events = normalize_events(raw, now)
    assert events == [
        {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 30},
        {"event": "PMI", "currency": "USD", "importance": "MEDIUM", "minutes_to_event": 60},
    ]


def test_normalize_events_skips_unparseable_or_past_dates():
    now = datetime(2026, 9, 16, 10, 0, tzinfo=timezone.utc)
    raw = [
        {"country": "US", "event": "past", "impact": "high",
         "date": "2026-09-16", "time": "09:00:00"},  # 60m ago
        {"country": "US", "event": "broken", "impact": "high",
         "date": "2026-09-16", "time": "not-a-time"},
        {"country": "US", "event": "nodate", "impact": "high", "date": ""},
    ]
    events = normalize_events(raw, now)
    assert [e["event"] for e in events] == ["past"]  # stored, gate filters it
    assert events[0]["minutes_to_event"] == -60


def test_blocking_events_importance_and_window():
    events = [
        {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 20},
        {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 45},
        {"event": "PMI", "currency": "USD", "importance": "MEDIUM", "minutes_to_event": 10},
        {"event": "past", "currency": "USD", "importance": "HIGH", "minutes_to_event": -5},
    ]
    assert [e["event"] for e in blocking_events(events, "HIGH", 30)] == ["CPI"]
    assert len(blocking_events(events, "MEDIUM", 30)) == 2  # CPI 20m + PMI 10m
    assert blocking_events(events, "HIGH", 15) == []


def test_finnhub_provider_fails_open_on_network_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("requests.get", boom)
    provider = FinnhubCalendarProvider("k")
    assert provider.upcoming_events(datetime.now(timezone.utc), 60) == []


def test_finnhub_provider_normalizes_and_filters(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"economicCalendar": [
                {"country": "US", "event": "CPI", "impact": "high",
                 "date": "2026-09-16", "time": "11:00:00"},
            ]}

    monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
    provider = FinnhubCalendarProvider("k")
    now = datetime(2026, 9, 16, 10, 30, tzinfo=timezone.utc)
    events = provider.upcoming_events(now, 60)
    assert events == [
        {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 30}
    ]


# ------------------------------------------------------------ risk gate


def _eval(settings: Settings, context: dict):
    engine = RiskEngine(settings)
    gates: list[dict] = []
    result = engine.evaluate(
        symbol="XAUUSD",
        timeframe="15m",
        side=Side.LONG,
        confidence=0.7,
        price=4350.0,
        atr=12.5,
        verdicts={},
        gauge=None,
        context=context,
        trail=gates,
    )
    return result, gates


def _settings(**overrides) -> Settings:
    return Settings(
        min_confidence=0.55,
        setup_quality_min=0.0,
        telegram_bot_token="",
        telegram_chat_id="",
        **overrides,
    )


def test_news_gate_blocks_high_importance_event_in_window():
    result, gates = _eval(
        _settings(news_filter_enabled=True, news_block_minutes=30),
        {"news_context": [
            {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 20}
        ]},
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.NEWS_RISK.value
    assert "CPI" in result.reason
    assert next(g for g in gates if g["gate"] == "news")["status"] == "reject"


def test_news_gate_passes_event_outside_window():
    result, gates = _eval(
        _settings(news_filter_enabled=True, news_block_minutes=30),
        {"news_context": [
            {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 45}
        ]},
    )
    assert not isinstance(result, Rejection)
    news_gate = next(g for g in gates if g["gate"] == "news")
    assert news_gate["status"] == "pass"


def test_news_gate_ignores_below_min_importance():
    result, _ = _eval(
        _settings(news_filter_enabled=True, news_block_minutes=30, news_min_importance="HIGH"),
        {"news_context": [
            {"event": "PMI", "currency": "USD", "importance": "MEDIUM", "minutes_to_event": 5}
        ]},
    )
    assert not isinstance(result, Rejection)


def test_news_gate_disabled_by_default():
    result, gates = _eval(
        _settings(),
        {"news_context": [
            {"event": "CPI", "currency": "USD", "importance": "HIGH", "minutes_to_event": 5}
        ]},
    )
    assert not isinstance(result, Rejection)
    assert not any(g["gate"] == "news" for g in gates)


# ------------------------------------------------------------- snapshot


class _FakeMarket:
    """Minimal market: candles only; calendar provider is injected."""

    def __init__(self, provider=None) -> None:
        self.calendar_provider = provider

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
        import pandas as pd

        n = 300
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="15min", tz="UTC")
        close = 4350.0 + pd.Series(range(n), dtype=float).to_numpy() * 0.01
        return pd.DataFrame(
            {"open": close, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 100.0},
            index=idx,
        )


class _StubCalendar:
    def upcoming_events(self, now: datetime, window_minutes: int) -> list[dict]:
        return [{"event": "CPI", "currency": "USD", "importance": "HIGH",
                 "minutes_to_event": 25}]


def test_snapshot_carries_news_and_shock_context():
    settings = Settings(
        snapshot_timeframes=["1h"], htf_timeframe="4h", htf_bias_filter_enabled=False,
        telegram_bot_token="", telegram_chat_id="",
    )
    snap = build_market_snapshot(_FakeMarket(provider=_StubCalendar()), "XAUUSD", settings, "15m")
    assert snap.news_context == [{"event": "CPI", "currency": "USD",
                                  "importance": "HIGH", "minutes_to_event": 25}]
    assert snap.shock_context["state"] == "NORMAL"
    assert snap.news_context == snap.entry_snapshot_for_llm()["news_context"]
    assert snap.context_for_risk()["news_context"] == snap.news_context
