"""V1 features: DXY concurrency hard gate, gold gauge, Telegram notifier.

The DXY rule is the heart of the user's strategy: LONG gold only when the
dollar is weak (gauge high), SHORT only when the dollar is strong (gauge
low). These tests pin that contract and the notification layer.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trading_agent.config import Settings
from trading_agent.data.gold import GoldData
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import AgentVerdict, Rejection, Side, SignalProposal


def verdicts() -> dict[str, AgentVerdict]:
    return {
        "technical": AgentVerdict(agent="technical", source="test", model="test", payload={}),
    }


def evaluate(settings: Settings, side: Side, gauge: dict) -> SignalProposal | Rejection:
    risk = RiskEngine(settings)
    return risk.evaluate(
        symbol="XAUUSD", timeframe="1h", side=side, confidence=0.7,
        price=4350.0, atr=3.5, verdicts=verdicts(), gauge=gauge,
    )


DXY_WEAK = {"value": 60, "classification": "Bullish (USD weak)", "kind": "dxy"}
DXY_STRONG = {"value": 40, "classification": "Bearish (USD strong)", "kind": "dxy"}


# ------------------------------------------------------------- DXY hard gate


def test_long_rejected_when_dollar_not_weak(base_settings, seeded):
    result = evaluate(base_settings, Side.LONG, DXY_STRONG)
    assert isinstance(result, Rejection)
    assert "DXY" in result.reason


def test_short_rejected_when_dollar_not_strong(base_settings, seeded):
    result = evaluate(base_settings, Side.SHORT, DXY_WEAK)
    assert isinstance(result, Rejection)
    assert "DXY" in result.reason


def test_long_passes_when_dollar_weak(base_settings, seeded):
    result = evaluate(base_settings, Side.LONG, DXY_WEAK)
    assert isinstance(result, SignalProposal)


def test_short_passes_when_dollar_strong(base_settings, seeded):
    result = evaluate(base_settings, Side.SHORT, DXY_STRONG)
    assert isinstance(result, SignalProposal)


def test_non_dxy_gauge_is_not_filtered(base_settings, seeded):
    fear_greed = {"value": 99, "classification": "Extreme Greed", "kind": "fear_greed"}
    result = evaluate(base_settings, Side.LONG, fear_greed)
    assert isinstance(result, SignalProposal)


def test_dxy_filter_can_be_disabled(base_settings, seeded):
    settings = base_settings.model_copy(update={"dxy_filter_enabled": False})
    result = evaluate(settings, Side.LONG, DXY_STRONG)
    assert isinstance(result, SignalProposal)


def test_missing_gauge_does_not_block(base_settings, seeded):
    result = evaluate(base_settings, Side.LONG, None)
    assert isinstance(result, SignalProposal)


# ------------------------------------------------------------- gold gauge


def _dxy_frame(days: int = 10, start: float = 100.0, step: float = -0.2) -> pd.DataFrame:
    idx = pd.date_range("2026-09-01", periods=days, freq="1D", tz="UTC")
    closes = [start + step * i for i in range(days)]
    return pd.DataFrame({"close": closes}, index=idx)


def test_gold_gauge_kind_tagged():
    gauge = GoldData._compute_gauge(_dxy_frame(), "test")
    assert gauge is not None
    assert gauge["kind"] == "dxy"
    # DXY falling 7 days -> dollar weak -> gauge above 50
    assert gauge["value"] > 50


def test_gold_gauge_short_history_returns_none():
    assert GoldData._compute_gauge(_dxy_frame(days=4), "test") is None


class _StubMT5:
    """Fake MT5 source that serves DXY candles; fails on demand."""

    def __init__(self, df: pd.DataFrame, fail: bool = False):
        self.df = df
        self.fail = fail
        self.calls: list[tuple] = []

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        self.calls.append((symbol, timeframe, limit))
        if self.fail:
            raise RuntimeError("mt5 down")
        return self.df


def test_gold_gauge_prefers_mt5_source():
    stub = _StubMT5(_dxy_frame(days=10, step=-0.5))
    gold = GoldData("binance", mt5=stub, dxy_symbol="DXY_U6")
    gauge = gold.sentiment_gauge()
    assert gauge is not None
    assert gauge["source"] == "MT5 DXY_U6"
    assert stub.calls == [("DXY_U6", "1h", 300)]


def test_gold_gauge_falls_back_to_yfinance_on_mt5_failure():
    stub = _StubMT5(_dxy_frame(), fail=True)
    gold = GoldData("binance", mt5=stub, dxy_symbol="DXY_U6")
    # yfinance may be unreachable in tests; the contract is: no crash and
    # either a gauge or None (never an exception).
    assert gold.sentiment_gauge() is None or gold.sentiment_gauge()["kind"] == "dxy"


# ------------------------------------------------------------- Telegram


@pytest.fixture
def telegram_settings() -> Settings:
    return Settings(telegram_bot_token="123:abc", telegram_chat_id="987")


def test_notifier_disabled_without_config(base_settings):
    assert not TelegramNotifier(base_settings).enabled


def test_notifier_sends_and_never_raises(telegram_settings, monkeypatch):
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self):
            return None

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr("httpx.post", fake_post)
    notifier = TelegramNotifier(telegram_settings)
    assert notifier.send("hello")
    assert captured["json"]["chat_id"] == "987"
    assert captured["json"]["text"] == "hello"
    assert captured["url"] == "https://api.telegram.org/bot123:abc/sendMessage"


def test_notifier_signal_contains_dxy_and_decision(telegram_settings, monkeypatch):
    sent: list[str] = []

    class FakeResp:
        def raise_for_status(self):
            return None

    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeResp())
    notifier = TelegramNotifier(telegram_settings)
    monkeypatch.setattr(notifier, "send", lambda text: sent.append(text) or True)
    proposal = SignalProposal(
        id="pid-1", symbol="XAUUSD", timeframe="1h", side=Side.LONG, confidence=0.72,
        entry=4350.0, stop=4300.0, target=4450.0, size=1.2, risk_amount=50.0,
        expected_rr=2.0, rationale="r", evidence={}, model="test",
    )
    notifier.send_signal(proposal, DXY_WEAK)
    text = sent[0]
    assert "LONG" in text
    assert "DXY : 60" in text
    assert "approve pid-1" in text


def test_notifier_network_failure_returns_false(telegram_settings, monkeypatch):
    def fake_post(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("httpx.post", fake_post)
    assert not TelegramNotifier(telegram_settings).send("x")
