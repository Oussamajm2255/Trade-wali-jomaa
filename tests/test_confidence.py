"""Confidence calibration infrastructure (spec §21).

raw_confidence stays abs(direction_score); calibrated_confidence is a
historical win rate inside the signal's own bucket and must be None
until enough resolved signal outcomes exist — the product must never
claim calibration early.

The read side (phase 5) is signal-based: evaluated_signal_outcomes
reads SignalRecord rows (fusion["raw_confidence"], outcome WIN/LOSS),
so these tests seed complete signal records.
"""
from __future__ import annotations

import itertools

import pytest

from trading_agent.config import Settings
from trading_agent.fusion.confidence import calibrated_confidence
from trading_agent.schema.types import utcnow
from trading_agent.store import actions

_seq = itertools.count(1)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        calibration_window=500,
        calibration_min_samples=50,
        calibration_min_per_bin=10,
    )


def _signal(confidence: float, outcome: str | None = None) -> str:
    """Seed one complete signal record; only WIN/LOSS outcomes are evaluated."""
    return actions.record_signal(
        {
            "signal_id": f"XAUUSD-20260101-{next(_seq):06d}",
            "ts": utcnow(),
            "symbol": "XAUUSD",
            "timeframe": "15m",
            "strategy_version": "LEGACY_BASELINE",
            "config_version": "1",
            "prompt_version": "TECH_V1/REGIME_V1/SENT_V1",
            "market_snapshot": {"price": 3000.0},
            "ai_outputs": {},
            "fusion": {"raw_confidence": confidence, "direction_score": confidence},
            "setup_quality": None,
            "conflicts": None,
            "gates": [{"gate": "final", "status": "pass"}],
            "final_decision": "proposal" if outcome in ("WIN", "LOSS") else "rejected",
            "outcome": outcome,
        }
    )


def test_no_data_means_no_calibration(settings: Settings) -> None:
    assert calibrated_confidence(0.7, settings) is None


def test_too_few_samples_means_no_calibration(settings: Settings) -> None:
    for _ in range(30):
        _signal(0.7, "WIN")
    for _ in range(19):
        _signal(0.7, "LOSS")
    assert calibrated_confidence(0.7, settings) is None


def test_calibrates_from_own_confidence_bucket(settings: Settings) -> None:
    # 50 resolved outcomes, 30 wins, all inside the [0.70, 0.75) bucket.
    for _ in range(30):
        _signal(0.70, "WIN")
    for _ in range(20):
        _signal(0.70, "LOSS")
    assert calibrated_confidence(0.72, settings) == 0.6
    assert calibrated_confidence(0.70, settings) == 0.6


def test_sparse_bucket_means_no_calibration(settings: Settings) -> None:
    # Enough resolved signals in total, but only 9 in the [0.70, 0.75) bucket.
    for _ in range(20):
        _signal(0.10, "WIN")
    for _ in range(21):
        _signal(0.10, "LOSS")
    for _ in range(9):
        _signal(0.70, "WIN")
    assert calibrated_confidence(0.70, settings) is None
    assert calibrated_confidence(0.10, settings) == pytest.approx(round(20 / 41, 4))


def test_unresolved_signals_never_count(settings: Settings) -> None:
    # 200 stored signals, but only 50 have resolved outcomes.
    for _ in range(200):
        _signal(0.70)
    for _ in range(25):
        _signal(0.70, "WIN")
    for _ in range(25):
        _signal(0.70, "LOSS")
    assert calibrated_confidence(0.70, settings) == 0.5


def test_ambiguous_outcomes_never_count(settings: Settings) -> None:
    # EXPIRED/BREAKEVEN/INVALIDATED and unresolved signals are stored but
    # never feed calibration — only WIN/LOSS are countable outcomes.
    for _ in range(9):
        _signal(0.70, "WIN")
    _signal(0.70, "EXPIRED")
    _signal(0.70)
    assert calibrated_confidence(0.70, settings) is None


def test_top_bucket_is_reachable_at_one(settings: Settings) -> None:
    # raw 1.0 lands in the [0.95, 1.0) bucket instead of its own empty one.
    for _ in range(40):
        _signal(0.96, "WIN")
    for _ in range(10):
        _signal(0.96, "LOSS")
    assert calibrated_confidence(1.0, settings) == 0.8
