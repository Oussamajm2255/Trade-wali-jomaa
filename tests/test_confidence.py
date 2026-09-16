"""Confidence calibration infrastructure (spec §21).

raw_confidence stays abs(direction_score); calibrated_confidence is a
historical win rate inside the signal's own bucket and must be None
until enough evaluated outcomes exist — the product must never claim
calibration early."""
from __future__ import annotations

import pytest

from trading_agent.config import Settings
from trading_agent.fusion.confidence import calibrated_confidence
from trading_agent.store import actions
from trading_agent.store.db import session_scope
from trading_agent.store.models import AgentTrack


@pytest.fixture
def settings() -> Settings:
    return Settings(
        calibration_window=500,
        calibration_min_samples=50,
        calibration_min_per_bin=10,
    )


def _track(confidence: float) -> dict:
    return {
        "agent": "technical",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "source": "llm",
        "model": "deepseek-chat",
        "prediction": {"bias": "long", "conviction": confidence},
        "market_regime": "trend_up",
        "confidence": confidence,
        "failure_reason": None,
        "fallback_method": None,
    }


def _set_outcomes(confidence: float, wins: int, losses: int) -> None:
    with session_scope() as session:
        rows = (
            session.query(AgentTrack)
            .filter(AgentTrack.confidence == confidence)
            .all()
        )
        for i, row in enumerate(rows):
            win = i < wins
            row.actual_outcome = "win" if win else "loss"
            row.correct = win


def test_no_data_means_no_calibration(settings: Settings) -> None:
    assert calibrated_confidence(0.7, settings) is None


def test_too_few_samples_means_no_calibration(settings: Settings) -> None:
    actions.record_agent_track([_track(0.7) for _ in range(49)])
    _set_outcomes(0.7, wins=30, losses=19)
    assert calibrated_confidence(0.7, settings) is None


def test_calibrates_from_own_confidence_bucket(settings: Settings) -> None:
    # 50 evaluated outcomes, 30 wins, all inside the [0.70, 0.75) bucket.
    actions.record_agent_track([_track(0.70) for _ in range(50)])
    _set_outcomes(0.70, wins=30, losses=20)
    assert calibrated_confidence(0.72, settings) == 0.6
    assert calibrated_confidence(0.70, settings) == 0.6


def test_sparse_bucket_means_no_calibration(settings: Settings) -> None:
    # Enough evaluated rows in total, but only 9 in the [0.70, 0.75) bucket.
    actions.record_agent_track([_track(0.10) for _ in range(41)])
    actions.record_agent_track([_track(0.70) for _ in range(9)])
    _set_outcomes(0.10, wins=20, losses=21)
    _set_outcomes(0.70, wins=9, losses=0)
    assert calibrated_confidence(0.70, settings) is None
    assert calibrated_confidence(0.10, settings) == pytest.approx(round(20 / 41, 4))


def test_unevaluated_rows_never_count(settings: Settings) -> None:
    # 200 tracked verdicts, but only 50 have outcomes.
    actions.record_agent_track([_track(0.70) for _ in range(200)])
    with session_scope() as session:
        rows = session.query(AgentTrack).limit(50).all()
        for i, row in enumerate(rows):
            row.actual_outcome = "win" if i < 25 else "loss"
            row.correct = i < 25
    assert calibrated_confidence(0.70, settings) == 0.5


def test_top_bucket_is_reachable_at_one(settings: Settings) -> None:
    # raw 1.0 lands in the [0.95, 1.0) bucket instead of its own empty one.
    actions.record_agent_track([_track(0.96) for _ in range(50)])
    _set_outcomes(0.96, wins=40, losses=10)
    assert calibrated_confidence(1.0, settings) == 0.8
