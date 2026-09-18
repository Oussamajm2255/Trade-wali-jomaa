"""Rejected-setup outcome analysis (V-MONSTER §67): did we refuse well?"""

from datetime import datetime, timedelta, timezone

from trading_agent.analytics.rejected import (
    RejectionOutcome,
    rejection_outcome,
    rejection_quality,
)
from trading_agent.dashboard.report import collect
from trading_agent.store import actions
from trading_agent.store.db import session_scope

NOW = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)


def _classify(ds, entry, future, threshold_pct=0.05):
    return rejection_outcome(ds, entry, future, threshold_pct=threshold_pct)


# ------------------------------------------------------ single classification


def test_correct_reject_when_market_went_against():
    # LONG was rejected; the market FELL -> refusal saved a loss.
    assert _classify(0.6, 4350.0, 4340.0) == RejectionOutcome.CORRECT_REJECT
    # SHORT was rejected; the market ROSE -> refusal saved a loss.
    assert _classify(-0.6, 4350.0, 4360.0) == RejectionOutcome.CORRECT_REJECT


def test_wrong_reject_when_market_moved_beyond_threshold():
    # LONG rejected, market rose 0.1% (>= 0.05%) -> the trade would have paid.
    assert _classify(0.6, 4350.0, 4350.0 * 1.001) == RejectionOutcome.WRONG_REJECT
    # SHORT rejected, market fell 0.1%.
    assert _classify(-0.6, 4350.0, 4350.0 * 0.999) == RejectionOutcome.WRONG_REJECT


def test_inconclusive_within_noise_band():
    # LONG rejected, market rose only 0.02% — noise, not signal.
    assert _classify(0.6, 4350.0, 4350.0 * 1.0002) == RejectionOutcome.INCONCLUSIVE


def test_threshold_band_edges():
    # Well above the 0.05% threshold -> decided (WRONG).
    assert _classify(0.6, 4350.0, 4350.0 * 1.001) == RejectionOutcome.WRONG_REJECT
    # Well inside the noise band -> INCONCLUSIVE.
    assert _classify(0.6, 4350.0, 4350.0 * 1.0001) == RejectionOutcome.INCONCLUSIVE


def test_custom_threshold_is_respected():
    assert _classify(0.6, 4350.0, 4350.0 * 1.001, threshold_pct=0.2) == (
        RejectionOutcome.INCONCLUSIVE
    )


def test_degenerate_inputs_are_inconclusive():
    assert _classify(0.0, 4350.0, 4350.0 * 1.01) == RejectionOutcome.INCONCLUSIVE
    assert _classify(0.6, 0.0, 4350.0) == RejectionOutcome.INCONCLUSIVE
    assert _classify(0.6, -5.0, 4350.0) == RejectionOutcome.INCONCLUSIVE


# ------------------------------------------------------------ aggregation


def test_quality_aggregates_and_honest_ratio():
    rows = [
        {"direction_score": 0.6, "entry_price": 4350.0, "future_price": 4340.0},  # correct
        {"direction_score": 0.6, "entry_price": 4350.0, "future_price": 4345.0},  # correct
        {"direction_score": 0.6, "entry_price": 4350.0, "future_price": 4400.0},  # wrong
        {"direction_score": 0.6, "entry_price": 4350.0, "future_price": 4351.0},  # inconclusive
    ]
    q = rejection_quality(rows)
    assert q["correct"] == 2
    assert q["wrong"] == 1
    assert q["inconclusive"] == 1
    assert q["resolved"] == 4
    # The inconclusive row never inflates the ratio (spec §36).
    assert q["correct_rate"] == round(2 / 3, 4)


def test_quality_empty_population_rate_is_none():
    q = rejection_quality([])
    assert q["correct"] == 0
    assert q["wrong"] == 0
    assert q["correct_rate"] is None


def test_quality_only_inconclusive_rate_is_none():
    rows = [{"direction_score": 0.6, "entry_price": 4350.0, "future_price": 4351.0}]
    q = rejection_quality(rows)
    assert q["inconclusive"] == 1
    assert q["correct_rate"] is None


# ------------------------------------------------------ dashboard pipeline


def _signal(signal_id, ts, ds, price):
    return {
        "signal_id": signal_id,
        "ts": ts,
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "config_version": "v1",
        "prompt_version": "p1",
        "market_snapshot": {"price": price},
        "ai_outputs": {},
        "fusion": {"direction_score": ds},
        "setup_quality": {"score": 0.7},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [],
        "final_decision": "rejected",
        "no_trade_reason": "DXY_FILTER",
        "outcome": None,
        "r_multiple": None,
    }


def test_rejection_outcomes_pairs_rejections_with_30m_snapshot():
    actions.record_signal(_signal("r1", NOW - timedelta(minutes=40), 0.6, 4350.0))
    actions.record_signal(_signal("r2", NOW - timedelta(minutes=40), -0.6, 4350.0))
    actions.record_signal(_signal("r3", NOW - timedelta(minutes=40), 0.0, 4350.0))
    # Only r1/r2 have a 30m snapshot; r3 has no directional score.
    actions.record_post_snapshot("r1", 30, 4340.0, 4345.0, 4335.0, now=NOW)
    actions.record_post_snapshot("r2", 30, 4390.0, 4395.0, 4385.0, now=NOW)
    rows = actions.rejection_outcomes()
    assert {r["signal_id"] for r in rows} == {"r1", "r2"}
    r1 = next(r for r in rows if r["signal_id"] == "r1")
    assert r1["direction_score"] == 0.6
    assert r1["entry_price"] == 4350.0
    assert r1["future_price"] == 4340.0


def test_dashboard_collect_includes_rejection_quality():
    actions.record_signal(_signal("r1", NOW - timedelta(minutes=40), 0.6, 4350.0))
    actions.record_signal(_signal("r2", NOW - timedelta(minutes=40), 0.6, 4350.0))
    actions.record_post_snapshot("r1", 30, 4340.0, 4345.0, 4335.0, now=NOW)  # correct
    actions.record_post_snapshot("r2", 30, 4400.0, 4405.0, 4395.0, now=NOW)  # wrong
    with session_scope() as session:
        data = collect(session)
    q = data["rejection_quality"]
    assert q["correct"] == 1
    assert q["wrong"] == 1
    assert q["correct_rate"] == 0.5
