"""Live signal supervision + human execution feedback (V-MONSTER §62/§63).

The guarantees under test:
- The per-tick classification of a pending proposal: VALID,
  DO_NOT_CHASE (price beyond the execution zone), INVALIDATED (stop or
  target reached), EXPIRED (actionability deadline passed) — with
  EXPIRED > INVALIDATED > DO_NOT_CHASE priority and honest boundaries.
- Persistence dedup: a state CHANGE writes once and is returned once;
  unchanged ticks are a no-op. INVALIDATED/EXPIRED move the proposal
  out of `pending` (approving a dead signal is refused).
- Telegram follow-ups fire on change only; the first None -> VALID
  classification is silent (the proposal message already says it).
- §63: the approve path stores the measured execution latency; its EMA
  overrides the Phase G reaction window (`user_reaction_seconds`).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.fusion.timing import compute_timing
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Side, SignalProposal
from trading_agent.store import actions
from trading_agent.store.db import session_scope
from trading_agent.store.models import Proposal
from trading_agent.supervision import SupervisionState, supervise_pending, supervise_proposal

from test_snapshot import FakeMarket, fresh_gauge

NOW = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)


# ----------------------------------------------------- pure state machine


def test_long_valid_entry_and_boundaries() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 100.5,
        max_chase=5.0, now=NOW,
    )
    assert out["state"] == SupervisionState.VALID.value
    # Price exactly at entry + chase is still inside the zone.
    edge = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 105.0, max_chase=5.0, now=NOW
    )
    assert edge["state"] == SupervisionState.VALID.value


def test_long_pullback_below_entry_still_valid() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 96.0, max_chase=5.0, now=NOW
    )
    assert out["state"] == SupervisionState.VALID.value


def test_long_do_not_chase_beyond_zone() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 105.5, max_chase=5.0, now=NOW
    )
    assert out["state"] == SupervisionState.DO_NOT_CHASE.value
    assert "execution zone" in out["detail"]


def test_long_stop_breach_invalidated() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 95.0, max_chase=5.0, now=NOW
    )
    assert out["state"] == SupervisionState.INVALIDATED.value
    assert "stop" in out["detail"]


def test_long_target_reached_invalidated() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 110.0, max_chase=5.0, now=NOW
    )
    assert out["state"] == SupervisionState.INVALIDATED.value
    assert "target" in out["detail"]


def test_short_mirrors_long() -> None:
    assert supervise_proposal(
        Side.SHORT, 100.0, 105.0, 90.0, 99.0, max_chase=5.0, now=NOW
    )["state"] == SupervisionState.VALID.value
    assert supervise_proposal(
        Side.SHORT, 100.0, 105.0, 90.0, 94.5, max_chase=5.0, now=NOW
    )["state"] == SupervisionState.DO_NOT_CHASE.value
    assert supervise_proposal(
        Side.SHORT, 100.0, 105.0, 90.0, 105.0, max_chase=5.0, now=NOW
    )["state"] == SupervisionState.INVALIDATED.value
    assert supervise_proposal(
        Side.SHORT, 100.0, 105.0, 90.0, 90.0, max_chase=5.0, now=NOW
    )["state"] == SupervisionState.INVALIDATED.value


def test_expired_beats_everything() -> None:
    past = NOW - timedelta(seconds=1)
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 95.0, deadline=past, max_chase=5.0, now=NOW
    )
    assert out["state"] == SupervisionState.EXPIRED.value
    assert "deadline" in out["detail"]


def test_no_deadline_never_expires() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 100.0, deadline=None, now=NOW
    )
    assert out["state"] == SupervisionState.VALID.value


def test_naive_deadline_treated_as_utc() -> None:
    naive_past = (NOW - timedelta(minutes=5)).replace(tzinfo=None)
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 100.0, deadline=naive_past, now=NOW
    )
    assert out["state"] == SupervisionState.EXPIRED.value


def test_chase_disabled_never_do_not_chase() -> None:
    out = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 500.0, max_chase=0.0, now=NOW
    )
    # Beyond target, so still INVALIDATED — but never DO_NOT_CHASE.
    assert out["state"] == SupervisionState.INVALIDATED.value
    out2 = supervise_proposal(
        Side.LONG, 100.0, 95.0, 110.0, 105.0, max_chase=0.0, now=NOW
    )
    assert out2["state"] == SupervisionState.VALID.value


def test_side_accepts_string_values() -> None:
    out = supervise_proposal("long", 100.0, 95.0, 110.0, 106.0, max_chase=5.0, now=NOW)
    assert out["state"] == SupervisionState.DO_NOT_CHASE.value
    out2 = supervise_proposal("short", 100.0, 105.0, 90.0, 94.0, max_chase=5.0, now=NOW)
    assert out2["state"] == SupervisionState.DO_NOT_CHASE.value


# --------------------------------------------------- persistence + dedup


def _proposal(**overrides) -> SignalProposal:
    base = dict(
        symbol="XAUUSD",
        timeframe="15m",
        side=Side.LONG,
        confidence=0.7,
        entry=4350.0,
        stop=4300.0,
        target=4450.0,
        size=1.0,
        risk_amount=50.0,
        expected_rr=2.0,
        rationale="r",
        evidence={},
        model="test",
    )
    base.update(overrides)
    return SignalProposal(**base)


def _row(proposal_id: str) -> Proposal:
    with session_scope() as session:
        return session.get(Proposal, proposal_id)  # type: ignore[return-value]


def test_update_supervision_change_then_stable() -> None:
    pid = actions.save_proposal(_proposal())
    # First classification: None -> VALID is a change.
    previous, changed = actions.update_supervision(pid, "VALID", "ok", now=NOW)
    assert (previous, changed) == (None, True)
    # Same state: a silent no-op — no write, no audit.
    previous, changed = actions.update_supervision(pid, "VALID", "still ok", now=NOW)
    assert (previous, changed) == ("VALID", False)
    assert _row(pid).supervision_detail == "ok"  # unchanged
    # Real transition.
    previous, changed = actions.update_supervision(
        pid, "DO_NOT_CHASE", "beyond zone", now=NOW
    )
    assert (previous, changed) == ("VALID", True)
    events = [a.event for a in actions.recent_audit()]
    assert events.count("proposal_supervised") == 2  # None->VALID + VALID->DO_NOT_CHASE


def test_terminal_state_expires_proposal_and_blocks_approval() -> None:
    pid = actions.save_proposal(_proposal())
    actions.update_supervision(pid, "EXPIRED", "deadline exceeded", terminal=True, now=NOW)
    row = _row(pid)
    assert row.status == "expired"
    assert "supervision" in (row.decision_note or "")
    assert actions.decide_proposal(pid, approve=True) is None


def test_do_not_chase_stays_pending() -> None:
    pid = actions.save_proposal(_proposal())
    actions.update_supervision(pid, "DO_NOT_CHASE", "beyond zone", now=NOW)
    assert _row(pid).status == "pending"
    assert actions.decide_proposal(pid, approve=True) is not None


def test_supervise_pending_notifies_only_on_real_change() -> None:
    pid = actions.save_proposal(_proposal())
    # First tick, valid price: classified but silent.
    assert supervise_pending("XAUUSD", 4350.5, now=NOW) == []
    assert _row(pid).supervision_state == SupervisionState.VALID.value
    # Same state again: still silent.
    assert supervise_pending("XAUUSD", 4351.0, now=NOW) == []
    # Price beyond the chase zone: one follow-up.
    pid2 = actions.save_proposal(_proposal(max_chase=2.0))
    supervise_pending("XAUUSD", 4350.5, now=NOW)  # classify silently
    events = supervise_pending("XAUUSD", 4353.0, now=NOW)
    assert len(events) == 1
    assert events[0]["proposal_id"] == pid2
    assert events[0]["state"] == SupervisionState.DO_NOT_CHASE.value
    assert events[0]["previous_state"] == SupervisionState.VALID.value
    # Dedup: the second tick with the same classification is silent.
    assert supervise_pending("XAUUSD", 4354.0, now=NOW) == []


def test_supervise_pending_invalidates_and_dedups_terminal() -> None:
    actions.save_proposal(_proposal(max_chase=0.0))
    supervise_pending("XAUUSD", 4350.5, now=NOW)
    events = supervise_pending("XAUUSD", 4299.0, now=NOW)  # below stop
    assert len(events) == 1
    assert events[0]["state"] == SupervisionState.INVALIDATED.value
    # The proposal is no longer pending -> nothing more to supervise.
    assert supervise_pending("XAUUSD", 4298.0, now=NOW) == []


# ------------------------------------------------------- latency feedback


def test_record_user_latency_and_ema() -> None:
    assert actions.user_latency_ema(span=3) is None  # no samples yet
    pid1 = actions.save_proposal(_proposal())
    pid2 = actions.save_proposal(_proposal())
    with session_scope() as session:
        for pid in (pid1, pid2):
            row = session.get(Proposal, pid)
            row.status = "approved"
            row.decided_at = NOW
        session.commit()
    actions.record_user_latency(pid1, 100.0)
    actions.record_user_latency(pid2, 200.0)
    # span 3 -> alpha 0.5: ema = 100 -> 0.5*200 + 0.5*100 = 150.
    assert actions.user_latency_ema(span=3) == 150.0
    assert _row(pid1).user_latency_s == 100.0


def test_timing_accepts_reaction_override() -> None:
    from test_timing import Snap, _timing_kwargs

    out = compute_timing(
        Side.LONG,
        Snap(),
        Settings(),
        **_timing_kwargs(user_reaction_seconds=10.0),
    )
    # 10s override + 3s Telegram latency.
    assert out["reaction_s"] == 13.0


# ------------------------------------------------------- orchestrator wiring


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


def _force_long(orch: Orchestrator, monkeypatch) -> None:
    from trading_agent.fusion.types import FusionResult

    monkeypatch.setattr(
        orch,
        "_fuse",
        lambda verdicts: FusionResult(side=Side.LONG, direction_score=0.8, raw_confidence=0.8),
    )


def _fake_timing(**overrides) -> dict:
    base = {
        "quality": 0.7,
        "components": {},
        "speed_state": "NORMAL",
        "pace_per_minute": 10.0,
        "lead_time_s": 300.0,
        "reaction_s": 123.0,
        "deadline_epoch": (NOW + timedelta(minutes=30)).timestamp(),
        "deadline_iso": None,
        "expected_price": 100.0,
        "expected_drift": 1.0,
        "max_chase": 0.5,
        "execution_zone": [99.5, 100.5],
        "room_r": 3.0,
        "too_late": False,
        "detail": "timing 0.70",
    }
    base.update(overrides)
    return base


def _patch_timing(monkeypatch, captured: dict) -> None:
    import trading_agent.agents.orchestrator as orch_module

    def fake(*a, **kw):
        captured["kwargs"] = kw
        return _fake_timing()

    monkeypatch.setattr(orch_module, "compute_timing", fake)


def test_orchestrator_stamps_deadline_and_chase_on_proposal(monkeypatch) -> None:
    captured: dict = {}
    _patch_timing(monkeypatch, captured)
    settings = _settings()
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    assert result.actionability_deadline is not None
    assert result.actionability_deadline.tzinfo is not None
    assert result.max_chase == 0.5
    # The persisted row carries the supervision inputs.
    pid = actions.save_proposal(result)
    row = _row(pid)
    assert row.deadline_at is not None
    assert row.max_chase == 0.5


def test_orchestrator_feeds_latency_ema_into_timing(monkeypatch) -> None:
    # Seed one approved proposal with a measured latency: EMA = 90s.
    pid = actions.save_proposal(_proposal())
    with session_scope() as session:
        row = session.get(Proposal, pid)
        row.status = "approved"
        row.decided_at = NOW
        session.commit()
    actions.record_user_latency(pid, 90.0)

    captured: dict = {}
    _patch_timing(monkeypatch, captured)
    settings = _settings()
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    assert captured["kwargs"]["user_reaction_seconds"] == 90.0


# -------------------------------------------------------------- telegram lines


def test_supervision_message_renders_all_states() -> None:
    notifier = TelegramNotifier(Settings(telegram_bot_token="1:a", telegram_chat_id="2"))
    for state, label in (
        ("VALID", "entrée toujours valable"),
        ("DO_NOT_CHASE", "ne pas chasser l'entrée"),
        ("INVALIDATED", "signal invalidé"),
        ("EXPIRED", "signal expiré"),
    ):
        text = notifier.supervision_message(
            {
                "proposal_id": "pid-1",
                "symbol": "XAUUSD",
                "state": state,
                "detail": "détail",
                "previous_state": "VALID",
            }
        )
        assert f"🔄 SUIVI XAUUSD : {label}" in text
        assert "Détail : détail" in text
        assert "État précédent : VALID" in text
        assert "Proposal : pid-1" in text


def test_supervision_message_without_previous_state() -> None:
    notifier = TelegramNotifier(Settings(telegram_bot_token="1:a", telegram_chat_id="2"))
    text = notifier.supervision_message(
        {"proposal_id": "p", "symbol": "XAUUSD", "state": "EXPIRED", "detail": "d"}
    )
    assert "État précédent" not in text
