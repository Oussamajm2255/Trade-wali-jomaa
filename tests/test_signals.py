"""Signal database (spec §22): unique IDs, complete records, gate trail.

Every cycle — proposal or rejection — is persisted with its snapshot,
AI outputs, fusion, setup quality, conflicts and gate trail, so any
historical decision stays fully explainable.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.risk.engine import RiskEngine
from trading_agent.store import actions
from trading_agent.store.models import AgentTrack


def _record(signal_id: str, **overrides) -> str:
    record = {
        "signal_id": signal_id,
        "ts": datetime.now(timezone.utc),
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "config_version": "1",
        "prompt_version": "TECH_V1/REGIME_V1/SENT_V1",
        "market_snapshot": {},
        "ai_outputs": {},
        "fusion": {},
        "setup_quality": None,
        "conflicts": None,
        "gates": [],
        "final_decision": "rejected",
        "decision_reason": "test",
    }
    record.update(overrides)
    return actions.record_signal(record)


# --- signal IDs -----------------------------------------------------------


def test_next_signal_id_format_and_sequence() -> None:
    ts = datetime(2026, 9, 16, 8, 30, tzinfo=timezone.utc)
    assert actions.next_signal_id("XAUUSD", ts) == "XAUUSD-20260916-000001"
    # Symbol punctuation is cleaned; the prefix is alphanumeric only.
    assert actions.next_signal_id("xau/usd", ts) == "XAUUSD-20260916-000001"
    _record("XAUUSD-20260916-000007")  # existing rows advance the sequence
    assert actions.next_signal_id("XAUUSD", ts) == "XAUUSD-20260916-000008"


def test_next_signal_id_restarts_each_day() -> None:
    d1 = datetime(2026, 9, 16, tzinfo=timezone.utc)
    d2 = datetime(2026, 9, 17, tzinfo=timezone.utc)
    assert actions.next_signal_id("XAUUSD", d1) == "XAUUSD-20260916-000001"
    assert actions.next_signal_id("XAUUSD", d2) == "XAUUSD-20260917-000001"


# --- record / list / link -------------------------------------------------


def test_record_and_get_signal() -> None:
    sid = _record(
        "XAUUSD-20260916-000001",
        fusion={"raw_confidence": 0.62},
        final_decision="proposal",
        sl=3000.0,
        tp=3010.0,
        size=1.0,
        risk_amount=100.0,
        gates=[{"gate": "final", "status": "pass", "detail": "ok"}],
    )
    row = actions.get_signal(sid)
    assert row is not None
    assert row.final_decision == "proposal"
    assert row.fusion["raw_confidence"] == 0.62
    assert row.sl == 3000.0 and row.tp == 3010.0
    assert row.gates[0]["status"] == "pass"
    assert actions.get_signal("missing") is None


def test_list_signals_filters() -> None:
    _record("XAUUSD-20260916-000001", final_decision="proposal")
    _record("XAUUSD-20260916-000002", final_decision="rejected", decision_reason="no trade")
    _record("BTCUSDT-20260916-000001", symbol="BTCUSDT", final_decision="proposal")
    assert len(actions.list_signals(limit=50)) == 3
    proposals = actions.list_signals(limit=50, decision="proposal")
    assert {r.signal_id for r in proposals} == {"XAUUSD-20260916-000001", "BTCUSDT-20260916-000001"}
    gold = actions.list_signals(limit=50, symbol="XAUUSD")
    assert len(gold) == 2


def test_link_signal_proposal() -> None:
    sid = _record("XAUUSD-20260916-000001", final_decision="proposal")
    actions.link_signal_proposal(sid, "proposal-uuid")
    assert actions.get_signal(sid).proposal_id == "proposal-uuid"
    actions.link_signal_proposal("missing-id", "x")  # never raises
    actions.link_signal_proposal(None, "x")  # never raises


def test_evaluated_signal_outcomes_reads_win_loss_only() -> None:
    _record("XAUUSD-20260916-000001", fusion={"raw_confidence": 0.7}, outcome="WIN")
    _record("XAUUSD-20260916-000002", fusion={"raw_confidence": 0.7}, outcome="LOSS")
    _record("XAUUSD-20260916-000003", fusion={"raw_confidence": 0.7}, outcome="BREAKEVEN")
    _record("XAUUSD-20260916-000004", fusion={"raw_confidence": 0.7})  # unresolved
    rows = actions.evaluated_signal_outcomes()
    assert len(rows) == 2
    assert {r["correct"] for r in rows} == {True, False}
    assert all(r["confidence"] == 0.7 for r in rows)


# --- orchestrator integration --------------------------------------------


class FakeMarket:
    """Deterministic market returning one synthetic rising frame."""

    def __init__(self, n: int = 320) -> None:
        self.last_source = "test"
        idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
        base = np.linspace(2500.0, 2800.0, n)
        self.df = pd.DataFrame(
            {
                "open": base,
                "high": base + 10.0,
                "low": base - 10.0,
                "close": base + 5.0,
                "volume": 100.0,
            },
            index=idx,
        )

    def fetch_ohlcv(self, symbol, timeframe="15m", limit=300, ttl=None):
        return self.df.tail(limit)

    def sentiment_gauge(self) -> dict:
        return {
            "value": 70,
            "classification": "Bullish (USD weak)",
            "source": "test",
            "ts": str(self.df.index[-1]),
            "kind": "dxy",
        }


class BrokenMarket(FakeMarket):
    def fetch_ohlcv(self, symbol, timeframe="15m", limit=300, ttl=None):
        from trading_agent.data.market import MarketDataError

        raise MarketDataError("feed down")


def _settings(**overrides) -> Settings:
    return Settings(
        timeframe="15m",
        ohlcv_limit=300,
        snapshot_timeframes=[],
        htf_bias_filter_enabled=False,
        dxy_filter_enabled=False,
        session_filter_enabled=False,
        data_quality_min_candles=5,
        data_quality_allow_degraded=True,
        setup_quality_min=0.0,
        conflict_block_conflicted=False,
        min_confidence=0.55,
        deepseek_api_key=None,
        telegram_bot_token="",
        telegram_chat_id="",
        **overrides,
    )


def test_run_full_records_complete_signal() -> None:
    settings = _settings()
    market = FakeMarket()
    orch = Orchestrator(settings, market, RiskEngine(settings))
    now = market.df.index[-1].to_pydatetime()
    result, verdicts, _entry, _gauge = orch.run_full("XAUUSD", "15m", now=now)

    # The frame's last candle closes on 2026-01-04 (320 x 15m from 2026-01-01).
    assert result.signal_id and result.signal_id.startswith("XAUUSD-20260104-")
    rows = actions.list_signals(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.signal_id == result.signal_id
    # SQLite returns naive datetimes; the stored ts is the replay time in UTC.
    assert pd.Timestamp(row.ts).tz_localize("UTC") == pd.Timestamp(now)
    assert row.final_decision in ("proposal", "rejected")
    assert row.strategy_version == "LEGACY_BASELINE"
    assert row.market_snapshot.get("price")
    assert isinstance(row.gates, list) and len(row.gates) >= 4
    assert set(row.ai_outputs) == set(verdicts)
    if row.final_decision == "proposal":
        assert row.sl is not None and row.tp is not None
        assert row.decision_reason is None
    else:
        assert row.decision_reason

    # Every verdict of the cycle is linked to the signal record.
    from sqlalchemy import select

    from trading_agent.store.db import session_scope

    with session_scope() as session:
        tracks = session.scalars(
            select(AgentTrack).where(AgentTrack.signal_id == row.signal_id)
        ).all()
    assert len(tracks) == len(verdicts)


def test_market_failure_still_records_rejection() -> None:
    settings = _settings()
    orch = Orchestrator(settings, BrokenMarket(), RiskEngine(settings))
    result, _v, _e, _g = orch.run_full("XAUUSD", "15m")
    from trading_agent.schema.types import Rejection

    assert isinstance(result, Rejection)
    rows = actions.list_signals(limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row.final_decision == "rejected"
    assert row.gates[0]["gate"] == "market_data"
    assert row.gates[0]["status"] == "reject"
    assert row.market_snapshot == {"error": "market data unavailable"}
