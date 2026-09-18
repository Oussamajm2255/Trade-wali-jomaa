"""Counterfactual entry timing (V-MONSTER §66): M1 what-if research."""

from datetime import datetime, timedelta, timezone

from trading_agent.analytics.counterfactual import (
    CF_OFFSETS_S,
    METHOD,
    counterfactual_entry,
    counterfactual_report,
    counterfactual_rr,
)
from trading_agent.schema.types import Side

BASE = datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)


def _bar(ts, o, c):
    return {"ts": ts, "open": o, "close": c}


def test_entry_offset_zero_stays_in_bar():
    bars = [_bar(BASE, 4350.0, 4352.0), _bar(BASE + timedelta(minutes=1), 4352.0, 4350.0)]
    cf = counterfactual_entry(bars, BASE + timedelta(seconds=30), 0)
    assert cf is not None
    assert cf["price"] == round(4350.0 + 2.0 * 0.5, 8)  # mid-bar interpolation
    assert cf["method"] == METHOD
    assert cf["bar_index"] == 0


def test_entry_positive_offset_shifts_bar():
    bars = [
        _bar(BASE, 4350.0, 4352.0),
        _bar(BASE + timedelta(minutes=1), 4352.0, 4356.0),
    ]
    # base 10:00:00 + 10s -> 10:00:10, fraction 10/60 of bar 0.
    cf = counterfactual_entry(bars, BASE, 10)
    assert cf is not None
    assert cf["price"] == round(4350.0 + 2.0 * (10 / 60), 8)


def test_entry_negative_offset_looks_back():
    bars = [_bar(BASE - timedelta(minutes=1), 4348.0, 4350.0), _bar(BASE, 4350.0, 4352.0)]
    # base 10:00:00 - 10s -> 09:59:50, fraction 50/60 of the previous bar.
    cf = counterfactual_entry(bars, BASE, -10)
    assert cf is not None
    assert cf["price"] == round(4348.0 + 2.0 * (50 / 60), 8)


def test_entry_none_on_data_gap():
    bars = [_bar(BASE + timedelta(minutes=2), 4350.0, 4352.0)]
    assert counterfactual_entry(bars, BASE, 0) is None
    assert counterfactual_entry([], BASE, 5) is None


def test_entry_none_when_shift_leaves_coverage():
    bars = [_bar(BASE, 4350.0, 4352.0)]
    assert counterfactual_entry(bars, BASE, 61) is None
    assert counterfactual_entry(bars, BASE, -1) is None


def test_entry_normalizes_naive_base_ts():
    bars = [_bar(BASE, 4350.0, 4352.0)]
    cf = counterfactual_entry(bars, BASE.replace(tzinfo=None), 0)
    assert cf is not None
    assert cf["ts"] == BASE


def test_rr_long_math_and_beyond_stop():
    assert counterfactual_rr(Side.LONG, 4350.0, 4340.0, 4380.0, 4352.0) == round(
        (4380.0 - 4352.0) / (4352.0 - 4340.0), 4
    )
    # Shifted entry below the stop makes the what-if nonsense.
    assert counterfactual_rr(Side.LONG, 4350.0, 4340.0, 4380.0, 4339.0) is None


def test_rr_short_math_and_beyond_stop():
    assert counterfactual_rr(Side.SHORT, 4350.0, 4360.0, 4320.0, 4348.0) == round(
        (4348.0 - 4320.0) / (4360.0 - 4348.0), 4
    )
    assert counterfactual_rr(Side.SHORT, 4350.0, 4360.0, 4320.0, 4361.0) is None
    assert counterfactual_rr(Side.NEUTRAL, 4350.0, 4340.0, 4380.0, 4352.0) is None


def test_report_rows_include_drift_and_rr():
    bars = [
        _bar(BASE - timedelta(minutes=1), 4348.0, 4350.0),
        _bar(BASE, 4350.0, 4352.0),
        _bar(BASE + timedelta(minutes=1), 4352.0, 4350.0),
    ]
    report = counterfactual_report(
        bars, BASE, Side.LONG, entry=4351.0, stop=4340.0, target=4380.0
    )
    assert len(report) == len(CF_OFFSETS_S)
    by_offset = {row["offset_s"]: row for row in report}
    row = by_offset[0]
    assert row["available"] is True
    assert row["drift"] == round(row["price"] - 4351.0, 8)
    assert row["rr"] is not None
    # Offset +10s falls in bar 0 too; every row must be honest about gaps.
    assert all(r["available"] is True for r in report)


def test_report_flags_missing_coverage():
    bars = [_bar(BASE, 4350.0, 4352.0)]
    report = counterfactual_report(
        bars, BASE, Side.LONG, entry=4351.0, stop=4340.0, target=4380.0
    )
    by_offset = {row["offset_s"]: row for row in report}
    assert by_offset[-10]["available"] is False
    assert by_offset[0]["available"] is True
