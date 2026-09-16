"""Monte Carlo risk analysis (spec §28): order randomization -> the
distributions of maximum drawdown, losing streaks, final equity and the
empirical risk of ruin."""
from __future__ import annotations

import pytest

from trading_agent.analytics.monte_carlo import run_monte_carlo


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError):
        run_monte_carlo([])


def test_same_seed_is_reproducible_and_other_seed_differs() -> None:
    rs = [1.0, -1.0, 0.5, -0.5] * 10
    a = run_monte_carlo(rs, n=200, seed=7)
    b = run_monte_carlo(rs, n=200, seed=7)
    assert a == b
    c = run_monte_carlo(rs, n=200, seed=8)
    assert a.final_equity != c.final_equity or a.max_drawdown_pct != c.max_drawdown_pct


def test_all_winners_have_zero_drawdown_and_zero_ruin() -> None:
    report = run_monte_carlo([1.0] * 50, n=100, seed=1)
    assert report.max_drawdown_pct == {"p5": 0.0, "p50": 0.0, "p95": 0.0}
    assert report.max_losing_streak["p50"] == 0.0
    assert report.risk_of_ruin == 0.0
    expected = 10_000.0 * (1.01**50)  # compounding at 1% risk per trade
    assert report.final_equity["p5"] == pytest.approx(expected, rel=1e-3)


def test_all_losers_ruin_and_streak_spans_the_sample() -> None:
    # 100 consecutive -1R trades at 1% risk: 0.99^100 ~ 0.37 of start.
    report = run_monte_carlo([-1.0] * 100, n=50, seed=2)
    assert report.risk_of_ruin == 1.0
    assert report.max_losing_streak["p5"] == 100.0
    assert report.max_losing_streak["p95"] == 100.0
    assert report.final_equity["p50"] < 10_000.0


def test_report_dict_has_the_spec_fields() -> None:
    report = run_monte_carlo([1.0, -1.0], n=10, seed=3)
    d = report.to_dict()
    assert set(d) == {
        "n_simulations",
        "seed",
        "risk_of_ruin",
        "max_drawdown_pct",
        "max_losing_streak",
        "final_equity",
    }
