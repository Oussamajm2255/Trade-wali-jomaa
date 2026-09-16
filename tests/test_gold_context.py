"""Gold-specific context (spec §10): key levels, distances, volatility."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from trading_agent.data.gold_context import compute_gold_context


def entry_df(days: int = 5) -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    idx = pd.date_range(end=end, periods=96 * days, freq="15min", tz="UTC")
    close = 2400.0 + np.arange(len(idx)) * 0.01
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 100.0},
        index=idx,
    )


def daily_df(days: int = 10) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("1D"), periods=days, freq="1D", tz="UTC")
    close = 2400.0 + np.arange(days) * 1.0
    return pd.DataFrame(
        {"open": close, "high": close + 5.0, "low": close - 5.0, "close": close, "volume": 100.0},
        index=idx,
    )


def test_daily_and_weekly_opens() -> None:
    ctx = compute_gold_context(entry_df(), daily_df())
    day_start = pd.Timestamp.now(tz="UTC").normalize()
    expected_open = entry_df()[entry_df().index >= day_start]["open"].iloc[0]
    assert ctx["daily_open"] == expected_open
    assert ctx["weekly_open"] is not None
    # Rising synthetic series; equality only in the day's first candle.
    assert ctx["price"] >= ctx["daily_open"]


def test_previous_day_levels() -> None:
    d = daily_df()
    ctx = compute_gold_context(entry_df(), d)
    assert ctx["prev_day_high"] == d["high"].iloc[-1]
    assert ctx["prev_day_low"] == d["low"].iloc[-1]
    assert ctx["prev_week_high"] is not None
    assert ctx["prev_week_low"] is not None


def test_session_high_low_within_window() -> None:
    df = entry_df()
    start = pd.Timestamp.now(tz="UTC") - timedelta(hours=2)
    ctx = compute_gold_context(df, None, session_start=start)
    window = df[df.index >= start]
    assert ctx["session_high"] == round(float(window["high"].max()), 8)
    assert ctx["session_low"] == round(float(window["low"].min()), 8)


def test_distances_and_volatility() -> None:
    ctx = compute_gold_context(entry_df(), daily_df())
    assert isinstance(ctx["distance_from_daily_open_pct"], float)
    assert ctx["distance_from_pdh_pct"] is not None
    assert 0.0 <= ctx["volatility_percentile_100"] <= 1.0


def test_no_daily_frame_degrades_levels_to_none() -> None:
    ctx = compute_gold_context(entry_df(), None)
    assert ctx["prev_day_high"] is None
    assert ctx["prev_day_low"] is None
    assert ctx["prev_week_high"] is None
