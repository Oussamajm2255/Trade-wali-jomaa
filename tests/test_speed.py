"""Market speed (V-MONSTER §27): classes, formation, acceleration, honesty."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.data.speed import SpeedState, _timeframe_minutes, compute_market_speed


def frame(
    n: int = 40,
    base_range: float = 1.0,
    last_range: float | None = None,
    ramp_from: int | None = None,
    ramp_range: float | None = None,
) -> pd.DataFrame:
    """Uniform 15m candles; optional bigger LAST candle and/or a ramp."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    close = np.full(n, 2400.0)
    rng = np.full(n, base_range)
    if last_range is not None:
        rng[-1] = last_range
    if ramp_from is not None and ramp_range is not None:
        rng[ramp_from:] = ramp_range
        if last_range is not None:
            rng[-1] = last_range
    open_ = close - rng / 2.0
    return pd.DataFrame(
        {"open": open_, "high": close + rng / 2.0, "low": close - rng / 2.0,
         "close": close, "volume": 100.0},
        index=idx,
    )


def test_timeframe_minutes_resolution() -> None:
    assert _timeframe_minutes("15m") == 15
    assert _timeframe_minutes("1h") == 60
    assert _timeframe_minutes("4h") == 240
    assert _timeframe_minutes("1d") == 1440
    assert _timeframe_minutes("2h") == 120  # unknown label -> digits * 60
    assert _timeframe_minutes("garbage") == 60  # unknown -> conservative


def test_uniform_market_is_normal() -> None:
    r = compute_market_speed(frame())
    assert r["state"] == SpeedState.NORMAL
    assert r["insufficient_history"] is False
    # Uniform 1.0-range candles on 15m: 1/15 per minute on both axes.
    assert r["range_per_minute"] == pytest.approx(1.0 / 15.0, rel=0.2)
    assert r["atr_per_minute"] == pytest.approx(1.0 / 15.0, rel=0.2)
    assert r["formation_ratio"] == pytest.approx(1.0, rel=0.1)
    assert r["acceleration"] == pytest.approx(1.0, rel=0.1)


def test_big_last_candle_is_fast() -> None:
    # The current (closed) candle consumed 2x the typical range.
    r = compute_market_speed(frame(last_range=2.0))
    assert r["formation_ratio"] == pytest.approx(2.0, rel=0.1)
    assert r["state"] == SpeedState.FAST  # 2.0 >= 1.8, accel ~1 -> not EXTREME


def test_extreme_range_is_extreme() -> None:
    r = compute_market_speed(frame(last_range=3.5))
    assert r["formation_ratio"] == pytest.approx(3.5, rel=0.1)
    assert r["state"] == SpeedState.EXTREME


def test_fast_with_acceleration_is_extreme() -> None:
    # A FAST candle on top of a fresh acceleration ramp -> EXTREME.
    r = compute_market_speed(
        frame(last_range=2.0), accel_lookback=3, accel_extreme_mult=1.05
    )
    # ATR lag sits on the flat 1.0 baseline; ATR now ticked up -> accel > 1.
    assert r["formation_ratio"] == pytest.approx(2.0, rel=0.1)
    assert r["acceleration"] > 1.0
    assert r["state"] == SpeedState.EXTREME
    # Without the accel lift the same candle is merely FAST.
    r2 = compute_market_speed(frame(last_range=2.0), accel_lookback=3)
    assert r2["state"] == SpeedState.FAST


def test_tiny_last_candle_is_slow() -> None:
    r = compute_market_speed(frame(last_range=0.3))
    assert r["formation_ratio"] == pytest.approx(0.3, rel=0.1)
    assert r["state"] == SpeedState.SLOW


def test_mid_candle_formation_scales_with_elapsed_time() -> None:
    df = frame(last_range=2.0)
    # Halfway through the candle (7.5 of 15 minutes): the same range
    # counts double — it was consumed twice as fast.
    now = df.index[-1] + pd.Timedelta(minutes=7.5)
    r = compute_market_speed(df, now=now)
    assert r["formation_ratio"] == pytest.approx(4.0, rel=0.1)  # 2.0 / 0.5
    assert r["state"] == SpeedState.EXTREME


def test_insufficient_history_fails_open_to_normal() -> None:
    r = compute_market_speed(frame(n=10))
    assert r["state"] == SpeedState.NORMAL
    assert r["insufficient_history"] is True
    assert r["range_ratio"] is None
    # No ATR baseline (degenerate candles) is also honest, never fast.
    flat = frame()
    flat["high"] = flat["low"]  # zero range
    r2 = compute_market_speed(flat)
    assert r2["state"] == SpeedState.NORMAL
    assert r2["insufficient_history"] is True


def test_classification_is_deterministic() -> None:
    assert compute_market_speed(frame(last_range=3.5)) == compute_market_speed(frame(last_range=3.5))
