"""Parameter sensitivity (spec §27): one isolated deterministic replay
per grid value, stable regions preferred over peak results."""
from __future__ import annotations

import pytest

from analytics_helpers import WARMUP, gold_frame, settings
from trading_agent.analytics.sensitivity import (
    SensitivityPoint,
    SensitivityReport,
    run_sensitivity,
)


def test_grid_points_match_values() -> None:
    gold = gold_frame(n=400)
    grid = {"min_confidence": [0.30, 0.40]}
    report = run_sensitivity(settings(), {"15m": gold}, grid, warmup=WARMUP)
    assert [p.parameter for p in report.points] == ["min_confidence", "min_confidence"]
    assert sorted(p.value for p in report.points) == [0.30, 0.40]
    assert all(p.stats.get("trades") is not None for p in report.points)


def test_unknown_parameter_raises() -> None:
    gold = gold_frame(n=400)
    with pytest.raises(ValueError, match="unknown parameter"):
        run_sensitivity(settings(), {"15m": gold}, {"not_a_knob": [1.0]}, warmup=WARMUP)


def test_sensitivity_replay_is_deterministic() -> None:
    gold = gold_frame(n=400)
    grid = {"take_profit_rr": [1.5, 2.0]}
    a = run_sensitivity(settings(), {"15m": gold}, grid, warmup=WARMUP)
    b = run_sensitivity(settings(), {"15m": gold}, grid, warmup=WARMUP)
    assert a.to_dict() == b.to_dict()


def test_stable_regions_prefer_plateaus_over_spikes() -> None:
    report = SensitivityReport(symbol="XAUUSD", timeframe="15m")
    # A steady 0.2 plateau with one isolated 0.5 spike at the end.
    for i, v in enumerate([0.2, 0.2, 0.2, 0.2, 0.5]):
        report.points.append(
            SensitivityPoint("min_confidence", float(i), {"expectancy_r": v})
        )
    spans = report.stable_regions("expectancy_r")["min_confidence"]
    assert len(spans) == 1
    assert spans[0]["start"] == 0.0
    assert spans[0]["end"] == 2.0
    # Stable first, magnitude second: the plateau wins, not the spike.
    best = report.best_stable("expectancy_r")
    assert best["min_confidence"] == pytest.approx(1.0)


def test_stable_regions_skip_unknown_metrics() -> None:
    report = SensitivityReport(symbol="XAUUSD", timeframe="15m")
    report.points.append(
        SensitivityPoint("min_confidence", 0.3, {"expectancy_r": None})
    )
    assert report.stable_regions("expectancy_r") == {"min_confidence": []}
    assert report.best_stable("expectancy_r") == {"min_confidence": None}
