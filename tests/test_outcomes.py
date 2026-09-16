"""Outcome engine (spec §23): classification + R/MFE/MAE/time metrics.

Win/loss alone is never the story: every closed position gets its R
multiple, MFE/MAE (price and R), bars held and time-to-stop/target, and
the outcome propagates to the signal record and the cycle's agent
tracks (the calibration read side, spec §21).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from trading_agent.outcome.engine import (
    Outcome,
    classify_outcome,
    excursion_r,
    finalize_position,
    position_metrics,
    r_multiple,
    time_to_exit,
)
from trading_agent.schema.types import Side
from trading_agent.store.db import session_scope
from trading_agent.store.models import AgentTrack, Position, Proposal, SignalRecord


def _position(**overrides) -> Position:
    fields = dict(
        proposal_id="proposal-1",
        symbol="XAUUSD",
        side=Side.LONG.value,
        size=10.0,
        entry=3000.0,
        stop=2990.0,
        target=3020.0,
        pnl=150.0,
        exit_reason="take_profit",
        exit_price=3020.0,
        mfe_price=3022.0,
        mae_price=2995.0,
        bars_open=12,
    )
    fields.update(overrides)
    return Position(**fields)


# --- classification -------------------------------------------------------


def test_classify_outcome_fixed_reasons() -> None:
    assert classify_outcome("expiry", 0.0) == Outcome.EXPIRED
    assert classify_outcome("invalidated", -50.0) == Outcome.INVALIDATED


def test_classify_outcome_by_pnl() -> None:
    assert classify_outcome("take_profit", 150.0) == Outcome.WIN
    assert classify_outcome("stop_loss", -100.0) == Outcome.LOSS
    assert classify_outcome("manual_close", 0.0) == Outcome.BREAKEVEN
    assert classify_outcome("stop_loss", None) is None


def test_trailing_stop_above_entry_is_a_win() -> None:
    # A trailing stop that closes above entry is a WIN even though the
    # exit reason is a stop (spec §23: PnL judges, not the reason).
    assert classify_outcome("stop_loss", 25.0) == Outcome.WIN


# --- metrics --------------------------------------------------------------


def test_r_multiple() -> None:
    # risk = |3000 - 2990| * 10 = 100; pnl 150 -> 1.5R
    assert r_multiple(150.0, 3000.0, 2990.0, 10.0) == 1.5
    assert r_multiple(-100.0, 3000.0, 2990.0, 10.0) == -1.0
    assert r_multiple(None, 3000.0, 2990.0, 10.0) is None
    assert r_multiple(150.0, 3000.0, 3000.0, 10.0) is None  # zero risk


def test_excursion_r_direction_aware() -> None:
    long_pos = _position(side=Side.LONG.value, entry=3000.0, stop=2990.0)
    assert excursion_r(long_pos, 3020.0) == 2.0
    assert excursion_r(long_pos, 2995.0) == -0.5
    short_pos = _position(side=Side.SHORT.value, entry=3000.0, stop=3010.0)
    assert excursion_r(short_pos, 2980.0) == 2.0
    assert excursion_r(short_pos, 3005.0) == -0.5


def test_time_to_exit() -> None:
    assert time_to_exit(_position(exit_reason="stop_loss", bars_open=7)) == (7, None)
    assert time_to_exit(_position(exit_reason="take_profit", bars_open=12)) == (None, 12)
    assert time_to_exit(_position(exit_reason="manual_close", bars_open=3)) == (None, None)


def test_position_metrics_shape() -> None:
    metrics = position_metrics(_position())
    assert metrics["outcome"] is None  # not finalised yet
    assert metrics["r_multiple"] is None
    assert metrics["mfe_r"] == pytest.approx(2.2)
    assert metrics["mae_r"] == pytest.approx(-0.5)
    assert metrics["bars_open"] == 12
    assert metrics["time_to_target"] == 12
    assert metrics["time_to_stop"] is None


# --- propagation ----------------------------------------------------------


def _seed_cycle(session, proposal_id: str, signal_id: str) -> Position:
    session.add(
        Proposal(
            id=proposal_id,
            symbol="XAUUSD",
            timeframe="15m",
            side="long",
            confidence=0.7,
            entry=3000.0,
            stop=2990.0,
            target=3020.0,
            size=10.0,
            risk_amount=100.0,
            expected_rr=2.0,
            rationale="test",
            evidence={},
            model="test",
            status="approved",
        )
    )
    session.add(
        SignalRecord(
            signal_id=signal_id,
            ts=datetime.now(timezone.utc),
            symbol="XAUUSD",
            timeframe="15m",
            strategy_version="LEGACY_BASELINE",
            config_version="1",
            prompt_version="TECH_V1",
            market_snapshot={},
            ai_outputs={},
            fusion={"raw_confidence": 0.7},
            gates=[],
            final_decision="proposal",
            proposal_id=proposal_id,
        )
    )
    session.add(
        AgentTrack(
            agent="technical",
            symbol="XAUUSD",
            timeframe="15m",
            source="fallback",
            model="heuristic-fallback",
            prediction={},
            signal_id=signal_id,
        )
    )
    session.add(
        AgentTrack(
            agent="dxy",
            symbol="XAUUSD",
            timeframe="15m",
            source="fallback",
            model="heuristic-fallback",
            prediction={},
            signal_id=signal_id,
        )
    )
    session.add(
        AgentTrack(
            agent="unrelated",
            symbol="XAUUSD",
            timeframe="15m",
            source="fallback",
            model="heuristic-fallback",
            prediction={},
            signal_id="another-signal",
        )
    )
    pos = _position(proposal_id=proposal_id)
    session.add(pos)
    session.flush()
    return pos


def test_finalize_position_propagates_win() -> None:
    with session_scope() as session:
        pos = _seed_cycle(session, "proposal-1", "XAUUSD-20260916-000001")
        finalize_position(session, pos)
        session.commit()
        assert pos.outcome == Outcome.WIN.value
        assert pos.r_multiple == pytest.approx(1.5)
    with session_scope() as session:
        sig = session.get(SignalRecord, "XAUUSD-20260916-000001")
        assert sig.outcome == "WIN"
        assert sig.r_multiple == pytest.approx(1.5)
        tracks = session.scalars(
            select(AgentTrack).where(AgentTrack.signal_id == "XAUUSD-20260916-000001")
        ).all()
        assert len(tracks) == 2
        assert all(t.actual_outcome == "WIN" and t.correct is True for t in tracks)
        unrelated = session.scalar(
            select(AgentTrack).where(AgentTrack.signal_id == "another-signal")
        )
        assert unrelated.actual_outcome is None and unrelated.correct is None


def test_finalize_position_loss_and_ambiguous() -> None:
    with session_scope() as session:
        pos = _seed_cycle(session, "proposal-2", "XAUUSD-20260916-000002")
        pos.pnl = -100.0
        pos.exit_reason = "stop_loss"
        finalize_position(session, pos)
        session.commit()
        assert pos.outcome == Outcome.LOSS.value
        assert pos.r_multiple == pytest.approx(-1.0)
    with session_scope() as session:
        sig = session.get(SignalRecord, "XAUUSD-20260916-000002")
        assert sig.outcome == "LOSS"
        tracks = session.scalars(
            select(AgentTrack).where(AgentTrack.signal_id == "XAUUSD-20260916-000002")
        ).all()
        assert all(t.correct is False for t in tracks)

    # EXPIRED: fixed classification, no correct/incorrect claim.
    with session_scope() as session:
        pos = _seed_cycle(session, "proposal-3", "XAUUSD-20260916-000003")
        pos.pnl = -5.0
        pos.exit_reason = "expiry"
        finalize_position(session, pos)
        session.commit()
        assert pos.outcome == Outcome.EXPIRED.value
    with session_scope() as session:
        tracks = session.scalars(
            select(AgentTrack).where(AgentTrack.signal_id == "XAUUSD-20260916-000003")
        ).all()
        assert all(t.correct is None for t in tracks)
