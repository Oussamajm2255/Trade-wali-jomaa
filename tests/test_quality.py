"""Data-quality validation states (spec §4): PASS / DEGRADED / FAIL."""
from __future__ import annotations

import pandas as pd

from trading_agent.data.quality import (
    DataQuality,
    QualityState,
    combine_quality,
    validate_candles,
    validate_gauge,
    validate_indicators,
)


def make_df(n: int = 300, timeframe: str = "15m", end: pd.Timestamp | None = None) -> pd.DataFrame:
    freq = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[timeframe]
    if end is None:
        end = pd.Timestamp.now(tz="UTC").floor(freq)
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    close = 2400.0 + pd.Series(range(n), dtype=float).to_numpy() * 0.01
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 100.0},
        index=idx,
    )


def fresh_gauge(**overrides) -> dict:
    gauge = {
        "value": 60,
        "classification": "Bullish (USD weak)",
        "kind": "dxy",
        "ts": str(pd.Timestamp.now(tz="UTC")),
    }
    gauge.update(overrides)
    return gauge


# ------------------------------------------------------------ candles

def test_valid_frame_passes() -> None:
    q = validate_candles(make_df(), "15m")
    assert q.state == QualityState.PASS
    assert q.issues == []


def test_empty_fails() -> None:
    q = validate_candles(pd.DataFrame(), "15m")
    assert q.state == QualityState.FAIL
    assert any("empty" in i for i in q.issues)


def test_unsorted_fails() -> None:
    q = validate_candles(make_df().iloc[::-1], "15m")
    assert q.state == QualityState.FAIL
    assert any("ordering" in i for i in q.issues)


def test_duplicates_degrades() -> None:
    df = pd.concat([make_df(), make_df().iloc[[-1]]])
    q = validate_candles(df, "15m")
    assert q.state == QualityState.DEGRADED
    assert any("duplicates" in i for i in q.issues)


def test_nan_fails() -> None:
    df = make_df()
    df.loc[df.index[5], "close"] = float("nan")
    q = validate_candles(df, "15m")
    assert q.state == QualityState.FAIL
    assert any("nan" in i for i in q.issues)


def test_non_positive_price_fails() -> None:
    df = make_df()
    df.loc[df.index[5], "close"] = 0.0
    q = validate_candles(df, "15m")
    assert q.state == QualityState.FAIL
    assert any("invalid_ohlc" in i for i in q.issues)


def test_high_below_low_fails() -> None:
    df = make_df()
    df.loc[df.index[5], "high"] = df.loc[df.index[5], "low"] - 1.0
    q = validate_candles(df, "15m")
    assert q.state == QualityState.FAIL
    assert any("high < low" in i for i in q.issues)


def test_too_few_candles_fails() -> None:
    q = validate_candles(make_df(n=10), "15m", min_candles=60)
    assert q.state == QualityState.FAIL
    assert any("min_candles" in i for i in q.issues)


def test_stale_frame_degrades() -> None:
    old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=12)
    q = validate_candles(make_df(end=old), "15m")
    assert q.state == QualityState.DEGRADED
    assert any("stale" in i for i in q.issues)


def test_fresh_frame_records_age_seconds() -> None:
    q = validate_candles(make_df(), "15m")
    assert q.age_s is not None and q.age_s >= 0


def test_age_anchored_to_replay_timestamp() -> None:
    replay_now = pd.Timestamp("2026-09-01 12:00", tz="UTC")
    df = make_df(end=replay_now - pd.Timedelta(minutes=5))
    q = validate_candles(df, "15m", now=replay_now)
    assert q.age_s == 300.0
    assert q.state == QualityState.PASS


def test_future_market_timestamp_flags_clock_skew() -> None:
    future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)
    q = validate_candles(make_df(end=future), "15m", clock_tolerance_s=60.0)
    assert q.state == QualityState.DEGRADED
    assert any("clock_skew" in i for i in q.issues)


def test_small_clock_skew_within_tolerance_passes() -> None:
    near_future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=30)
    q = validate_candles(make_df(end=near_future), "15m", clock_tolerance_s=60.0)
    assert q.state == QualityState.PASS


# ------------------------------------------------------------ gauge

def test_missing_gauge_degrades() -> None:
    q = validate_gauge(None)
    assert q.state == QualityState.DEGRADED
    assert any("unavailable" in i for i in q.issues)


def test_stale_gauge_degrades() -> None:
    old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=72)
    q = validate_gauge(fresh_gauge(ts=str(old)))
    assert q.state == QualityState.DEGRADED
    assert any("stale" in i for i in q.issues)


def test_fresh_gauge_passes() -> None:
    q = validate_gauge(fresh_gauge())
    assert q.state == QualityState.PASS


def test_non_dxy_gauge_not_checked() -> None:
    q = validate_gauge({"value": 99, "classification": "Extreme Greed", "kind": "fear_greed"})
    assert q.state == QualityState.PASS


# ------------------------------------------------------------ indicators

def test_invalid_atr_fails() -> None:
    q = validate_indicators({"atr_14": 0.0})
    assert q.state == QualityState.FAIL
    assert any("atr" in i for i in q.issues)


def test_missing_atr_fails() -> None:
    q = validate_indicators({})
    assert q.state == QualityState.FAIL


def test_valid_atr_passes() -> None:
    assert validate_indicators({"atr_14": 3.5}).state == QualityState.PASS


# ------------------------------------------------------------ combine

def test_combine_worst_state_wins() -> None:
    fail = DataQuality(state=QualityState.FAIL, issues=["a"], checks={"x": "fail"})
    degraded = DataQuality(state=QualityState.DEGRADED, issues=["b"], checks={"y": "degraded"})
    out = combine_quality(fail, degraded)
    assert out.state == QualityState.FAIL
    assert out.issues == ["a", "b"]
    assert out.checks == {"x": "fail", "y": "degraded"}


def test_combine_passes_with_degrades() -> None:
    out = combine_quality(
        DataQuality(),
        DataQuality(state=QualityState.DEGRADED, issues=["c"]),
    )
    assert out.state == QualityState.DEGRADED
    assert out.issues == ["c"]
