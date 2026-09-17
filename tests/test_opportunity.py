"""Opportunity clustering + signal dedup (V-MONSTER §31/§40/§41/§52/§53).

The guarantees under test:
- The OPPORTUNITY_ID is deterministic: same snapshot anchor -> same
  identity, side-scoped, and None without an anchor (never blocks).
- The dedup verdict: recent proposal at a close price -> SIGNAL_DUPLICATE;
  outside the window or at a distant price -> allowed; a pending stronger
  signal suppresses a weaker re-signal (OPPORTUNITY_ACTIVE) while a
  stronger one supersedes; disabled config, no anchor or a DB failure
  never blocks.
- The lifecycle: FORMING -> TRIGGERED -> EXPIRED (TTL), with TRIGGERED
  never downgraded by later non-triggering cycles.
- The orchestrator gate: wired after fusion, records the verdict in the
  gate trail, stamps signal records with the opportunity identity, and
  upserts the lifecycle row.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.fusion.types import FusionResult, NoTradeReason
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, Side, SignalProposal
from trading_agent.store import opportunity
from trading_agent.store.db import session_scope
from trading_agent.store.models import Opportunity, Proposal, SignalRecord

from test_snapshot import FakeMarket, fresh_gauge


class Snap:
    def __init__(self, structure=None, liquidity=None, price: float = 2000.0) -> None:
        self.structure = structure or {}
        self.liquidity = liquidity or {}
        self.price = price


def _settings(**overrides) -> Settings:
    base = dict(
        htf_timeframe="4h",
        snapshot_timeframes=["1h"],
        min_confidence=0.0,
        setup_quality_min=0.0,
        conflict_block_conflicted=False,
        htf_bias_filter_enabled=False,
        room_gate_enabled=False,
        deepseek_api_key=None,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    base.update(overrides)
    return Settings(**base)


# --- anchor + identity --------------------------------------------------


def test_neutral_side_has_no_anchor() -> None:
    assert opportunity.anchor_from_snapshot(Snap(), Side.NEUTRAL) is None


def test_bos_anchor_prefers_most_recent() -> None:
    snap = Snap(
        structure={
            "bos": [
                {"type": "BOS_BULLISH", "age": 5, "timestamp": "2026-01-01T10:00:00"},
                {"type": "BOS_BULLISH", "age": 1, "timestamp": "2026-01-01T11:00:00"},
                {"type": "BOS_BEARISH", "age": 0, "timestamp": "2026-01-01T11:30:00"},
            ]
        }
    )
    assert opportunity.anchor_from_snapshot(snap, Side.LONG) == "bos:2026-01-01T11:00:00"
    assert opportunity.anchor_from_snapshot(snap, Side.SHORT) == "bos:2026-01-01T11:30:00"


def test_displacement_anchor_falls_back() -> None:
    snap = Snap(
        structure={
            "displacements": [
                {"direction": "bearish", "age": 2, "timestamp": "t1"},
                {"direction": "bullish", "age": 1, "timestamp": "t2"},
            ]
        }
    )
    assert opportunity.anchor_from_snapshot(snap, Side.LONG) == "disp:t2"


def test_liquidity_anchor_falls_back() -> None:
    snap = Snap(
        liquidity={
            "levels": [
                {"price": 1990.0, "kind": "PDL"},
                {"price": 2010.0, "kind": "PDH"},
                {"price": 2005.0, "kind": "EQH"},
            ]
        },
        price=2000.0,
    )
    # Trigger-side pool: lows for LONG, highs for SHORT.
    assert opportunity.anchor_from_snapshot(snap, Side.LONG) == "liq:1990.0"
    assert opportunity.anchor_from_snapshot(snap, Side.SHORT) == "liq:2005.0"


def test_no_data_means_no_anchor() -> None:
    assert opportunity.anchor_from_snapshot(Snap(), Side.LONG) is None


def test_opportunity_id_stable_and_side_scoped() -> None:
    oid = opportunity.opportunity_id("xauusd", "15m", Side.LONG, "bos:ts")
    assert oid == "XAUUSD:15m:long:bos:ts"
    assert oid == opportunity.opportunity_id("xauusd", "15m", Side.LONG, "bos:ts")
    assert oid != opportunity.opportunity_id("xauusd", "15m", Side.SHORT, "bos:ts")
    assert opportunity.opportunity_id("xauusd", "15m", Side.LONG, None) is None


# --- dedup verdict ------------------------------------------------------


def _seed_signal(
    oid: str,
    *,
    final: str = "proposal",
    price: float = 2000.0,
    ts: datetime | None = None,
    setup: float = 0.8,
    proposal_id: str | None = None,
) -> str:
    with session_scope() as session:
        row = SignalRecord(
            signal_id=f"SIG{uuid4().hex[:16]}",
            ts=ts or datetime.now(timezone.utc),
            symbol="XAUUSD",
            timeframe="15m",
            strategy_version="t",
            config_version="1",
            prompt_version="p",
            market_snapshot={"price": price},
            ai_outputs={},
            fusion={},
            setup_quality={"score": setup},
            conflicts={},
            gates=[],
            final_decision=final,
            opportunity_id=oid,
            proposal_id=proposal_id,
        )
        session.add(row)
        return row.signal_id


def _seed_pending(oid: str, setup: float) -> None:
    with session_scope() as session:
        p = Proposal(
            id=str(uuid4()),
            symbol="XAUUSD",
            timeframe="15m",
            side="long",
            confidence=0.8,
            entry=2000.0,
            stop=1990.0,
            target=2020.0,
            size=0.1,
            risk_amount=10.0,
            expected_rr=2.0,
            rationale="t",
            evidence={},
            model="test",
            status="pending",
        )
        session.add(p)
        session.flush()
        session.add(
            SignalRecord(
                signal_id=f"SIG{uuid4().hex[:16]}",
                ts=datetime.now(timezone.utc),
                symbol="XAUUSD",
                timeframe="15m",
                strategy_version="t",
                config_version="1",
                prompt_version="p",
                market_snapshot={"price": 2000.0},
                ai_outputs={},
                fusion={},
                setup_quality={"score": setup},
                conflicts={},
                gates=[],
                final_decision="proposal",
                opportunity_id=oid,
                proposal_id=p.id,
            )
        )


def _verdict(
    anchor: str | None = "bos:t",
    price: float = 2000.0,
    setup: float = 0.8,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> dict | None:
    return opportunity.dedup_verdict(
        "XAUUSD",
        "15m",
        Side.LONG,
        anchor,
        price,
        setup,
        settings or Settings(opportunity_dedup_enabled=True),
        now=now,
    )


def test_dedup_disabled_or_no_anchor_never_blocks() -> None:
    assert _verdict(settings=Settings(opportunity_dedup_enabled=False)) is None
    assert _verdict(anchor=None) is None


def test_recent_proposal_at_close_price_is_duplicate() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    now = datetime.now(timezone.utc)
    _seed_signal(oid, ts=now - timedelta(minutes=10), price=2000.0)
    verdict = _verdict(price=2000.1, now=now)
    assert verdict is not None
    assert verdict["no_trade_reason"] == NoTradeReason.SIGNAL_DUPLICATE.value
    assert "already signalled" in verdict["detail"]


def test_duplicate_outside_window_is_allowed() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    now = datetime.now(timezone.utc)
    _seed_signal(oid, ts=now - timedelta(hours=4), price=2000.0)
    assert _verdict(now=now) is None  # 4h > 180m dedup window


def test_duplicate_at_distant_price_is_allowed() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    now = datetime.now(timezone.utc)
    _seed_signal(oid, ts=now - timedelta(minutes=5), price=2000.0)
    assert _verdict(price=2050.0, now=now) is None  # 2.5% > 0.1% tolerance


def test_different_opportunity_never_blocks() -> None:
    other = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:other")
    now = datetime.now(timezone.utc)
    _seed_signal(other, ts=now - timedelta(minutes=5), price=2000.0)
    assert _verdict(anchor="bos:t", now=now) is None


def test_pending_stronger_suppresses_weaker() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    _seed_pending(oid, setup=0.9)
    verdict = _verdict(setup=0.6)
    assert verdict is not None
    assert verdict["no_trade_reason"] == NoTradeReason.OPPORTUNITY_ACTIVE.value
    assert "stronger" in verdict["detail"]


def test_stronger_signal_supersedes_pending() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    _seed_pending(oid, setup=0.6)
    assert _verdict(setup=0.9) is None


def test_rejected_cycles_never_block() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    now = datetime.now(timezone.utc)
    _seed_signal(oid, final="rejected", ts=now - timedelta(minutes=5), price=2000.0)
    assert _verdict(now=now) is None


def test_db_failure_fails_open(monkeypatch) -> None:
    def boom():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    monkeypatch.setattr(opportunity, "session_scope", boom)
    assert _verdict() is None


# --- lifecycle ----------------------------------------------------------


def _track(oid: str, anchor: str, triggered: bool, now: datetime, settings: Settings) -> None:
    opportunity.track_opportunity(
        oid, "XAUUSD", "15m", Side.LONG, anchor,
        triggered=triggered, trigger_signal_id="SIG1" if triggered else None,
        settings=settings, now=now,
    )


def test_lifecycle_form_to_triggered() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    settings = Settings()
    now = datetime.now(timezone.utc)
    _track(oid, "bos:t", triggered=False, now=now, settings=settings)
    with session_scope() as session:
        assert session.get(Opportunity, oid).state == "FORMING"
    _track(oid, "bos:t", triggered=True, now=now, settings=settings)
    with session_scope() as session:
        row = session.get(Opportunity, oid)
        assert row.state == "TRIGGERED"
        assert row.trigger_signal_id == "SIG1"
    # A later non-triggering cycle must not downgrade a TRIGGERED one.
    _track(oid, "bos:t", triggered=False, now=now + timedelta(minutes=30), settings=settings)
    with session_scope() as session:
        assert session.get(Opportunity, oid).state == "TRIGGERED"


def test_lifecycle_expires_after_ttl() -> None:
    oid = opportunity.opportunity_id("XAUUSD", "15m", Side.LONG, "bos:t")
    settings = Settings(opportunity_ttl_minutes=720)
    t0 = datetime.now(timezone.utc)
    _track(oid, "bos:t", triggered=False, now=t0, settings=settings)
    _track(oid, "bos:t", triggered=False, now=t0 + timedelta(hours=13), settings=settings)
    with session_scope() as session:
        assert session.get(Opportunity, oid).state == "EXPIRED"


# --- orchestrator wiring -------------------------------------------------


def _orch(settings: Settings) -> Orchestrator:
    market = FakeMarket(gauge=fresh_gauge())
    return Orchestrator(settings, market, RiskEngine(settings))


def _force_long(orch: Orchestrator, monkeypatch) -> None:
    monkeypatch.setattr(
        orch,
        "_fuse",
        lambda verdicts: FusionResult(side=Side.LONG, direction_score=0.8, raw_confidence=0.8),
    )


def test_orchestrator_gate_suppresses_duplicate(monkeypatch) -> None:
    settings = _settings()
    orch = _orch(settings)
    _force_long(orch, monkeypatch)
    monkeypatch.setattr(
        opportunity, "anchor_from_snapshot", lambda snap, side: "bos:2026-01-01T00:00:00"
    )
    monkeypatch.setattr(
        opportunity,
        "dedup_verdict",
        lambda *a, **kw: {
            "no_trade_reason": NoTradeReason.SIGNAL_DUPLICATE.value,
            "detail": "opportunity already signalled",
        },
    )
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.SIGNAL_DUPLICATE.value
    assert "opportunity duplicate" in result.reason
    oid = "XAUUSD:15m:long:bos:2026-01-01T00:00:00"
    with session_scope() as session:
        rows = list(session.scalars(select(SignalRecord)))
        assert len(rows) == 1
        assert rows[0].opportunity_id == oid
        assert rows[0].opportunity_state == "FORMING"
        opp_gates = [g for g in (rows[0].gates or []) if g["gate"] == "opportunity"]
        assert opp_gates == [
            {"gate": "opportunity", "status": "reject", "detail": "opportunity already signalled"}
        ]
        assert session.get(Opportunity, oid).state == "FORMING"


def test_orchestrator_records_triggered_opportunity(monkeypatch) -> None:
    settings = _settings()
    orch = _orch(settings)
    _force_long(orch, monkeypatch)
    monkeypatch.setattr(opportunity, "anchor_from_snapshot", lambda snap, side: "bos:T")
    calls = {"n": 0}

    def _pass(*a, **kw):
        calls["n"] += 1
        return None

    monkeypatch.setattr(opportunity, "dedup_verdict", _pass)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    assert calls["n"] == 1
    oid = "XAUUSD:15m:long:bos:T"
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        assert row.opportunity_id == oid
        assert row.opportunity_state == "TRIGGERED"
        opp = session.get(Opportunity, oid)
        assert opp.state == "TRIGGERED"
        assert opp.trigger_signal_id == row.signal_id
        opp_gates = [g for g in (row.gates or []) if g["gate"] == "opportunity"]
        assert opp_gates == [{"gate": "opportunity", "status": "pass", "detail": oid}]


def test_orchestrator_dedup_disabled_never_checks(monkeypatch) -> None:
    settings = _settings(opportunity_dedup_enabled=False)
    orch = _orch(settings)
    _force_long(orch, monkeypatch)
    calls = {"n": 0}

    def _never(*a, **kw):
        calls["n"] += 1
        return {"no_trade_reason": "SIGNAL_DUPLICATE", "detail": "x"}

    monkeypatch.setattr(opportunity, "dedup_verdict", _never)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    assert calls["n"] == 0
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        # Identity stamping is forensics, not the gate: even with dedup
        # disabled the record carries the clustered opportunity.
        assert row.opportunity_id is not None
        assert row.opportunity_state == "TRIGGERED"
        assert session.get(Opportunity, row.opportunity_id).state == "TRIGGERED"
