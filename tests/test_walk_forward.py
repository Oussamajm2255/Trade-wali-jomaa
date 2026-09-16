"""Walk-forward validation (spec §26): TRAIN -> VALIDATION -> OUT-OF-
SAMPLE windows replayed on isolated state with no cross-window leakage."""
from __future__ import annotations

import pytest

from analytics_helpers import WARMUP, gold_frame, settings
from trading_agent.analytics.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
)


def test_window_geometry_is_exact() -> None:
    gold = gold_frame(n=900)
    config = WalkForwardConfig(train_bars=WARMUP, val_bars=50, test_bars=100, step_bars=200)
    report = run_walk_forward(settings(), {"15m": gold}, config, warmup=WARMUP)
    assert len(report.windows) == 2
    # Window 0: warmup ends at 250; train = idx 250..499; val = 500..549;
    # test = 550..649 (absolute positions in the frame).
    w = report.windows[0]
    assert w.train_start == str(gold.index[250])
    assert w.train_end == str(gold.index[499])
    assert w.val_start == str(gold.index[500])
    assert w.val_end == str(gold.index[549])
    assert w.test_start == str(gold.index[550])
    assert w.test_end == str(gold.index[649])
    assert w.candles == 100
    # Consecutive windows advance by step_bars and never overlap.
    assert report.windows[1].test_start == str(gold.index[750])
    assert report.windows[1].test_end == str(gold.index[849])


def test_aggregate_is_consistent_with_windows() -> None:
    gold = gold_frame(n=900)
    config = WalkForwardConfig(train_bars=WARMUP, val_bars=0, test_bars=100)
    report = run_walk_forward(settings(), {"15m": gold}, config, warmup=WARMUP)
    agg = report.aggregate
    assert agg["windows"] == len(report.windows) == 4
    assert agg["total_trades"] == sum(w.stats.get("trades", 0) for w in report.windows)
    assert agg["profitable_windows"] == sum(
        1 for w in report.windows if (w.stats.get("total_r") or 0) > 0
    )
    assert 0.0 <= agg["profitable_ratio"] <= 1.0
    # Expectancy aggregates only exist over windows that traded.
    traded = [w for w in report.windows if w.stats.get("expectancy_r") is not None]
    if traded:
        assert agg["avg_expectancy_r"] is not None
        assert agg["stdev_expectancy_r"] >= 0.0


def test_walk_forward_is_deterministic() -> None:
    gold = gold_frame(n=600)
    config = WalkForwardConfig(train_bars=WARMUP, val_bars=0, test_bars=50)
    a = run_walk_forward(settings(), {"15m": gold}, config, warmup=WARMUP)
    b = run_walk_forward(settings(), {"15m": gold}, config, warmup=WARMUP)
    assert a.to_dict() == b.to_dict()


def test_train_bars_shorter_than_warmup_raises() -> None:
    gold = gold_frame(n=900)
    config = WalkForwardConfig(train_bars=WARMUP - 1, test_bars=100)
    with pytest.raises(ValueError, match="warmup"):
        run_walk_forward(settings(), {"15m": gold}, config, warmup=WARMUP)


def test_frame_too_short_for_one_window_raises() -> None:
    gold = gold_frame(n=300)
    config = WalkForwardConfig(train_bars=WARMUP, test_bars=100)
    with pytest.raises(ValueError, match="need at least"):
        run_walk_forward(settings(), {"15m": gold}, config, warmup=WARMUP)


def test_missing_entry_frame_raises() -> None:
    config = WalkForwardConfig(train_bars=WARMUP, test_bars=100)
    with pytest.raises(ValueError, match="no historical frame"):
        run_walk_forward(settings(), {}, config, warmup=WARMUP)


def test_invalid_geometry_raises() -> None:
    with pytest.raises(ValueError):
        WalkForwardConfig(train_bars=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(test_bars=0)
    with pytest.raises(ValueError):
        WalkForwardConfig(val_bars=-1)
    with pytest.raises(ValueError):
        WalkForwardConfig(step_bars=0)
