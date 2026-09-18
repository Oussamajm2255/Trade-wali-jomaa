"""VWAP engine (V-MONSTER §12): math, trust gating, cross state, trend."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from trading_agent.data.vwap import compute_vwap

DAY = "2026-01-05"


def flat_df(closes: list[float], freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range(start=DAY, periods=len(closes), freq=freq, tz="UTC")
    close = pd.Series(closes, dtype=float, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 100.0}
    )


def test_vwap_is_typical_price_volume_mean() -> None:
    # tp == close here (high/low symmetric); equal volume -> plain mean.
    df = flat_df([100.0, 101.0, 102.0, 110.0])
    out = compute_vwap(df)
    assert out["available"] is True
    assert out["daily_vwap"] == pytest.approx((100 + 101 + 102 + 110) / 4)
    assert out["session_vwap"] == out["daily_vwap"]  # no session_start -> day window
    assert out["distance_to_daily_pct"] == pytest.approx(6.538, abs=0.001)


def test_volume_basis_labelled_per_source() -> None:
    """Tick-volume VWAP (MT5) is available but labelled, never silent."""
    df = flat_df([100.0, 101.0, 102.0, 110.0])
    tick = compute_vwap(df, volume_basis="tick")
    assert tick["available"] is True
    assert tick["volume_basis"] == "tick"
    real = compute_vwap(df)
    assert real["volume_basis"] == "real"
    proxy = compute_vwap(df, trust_volume=False, volume_basis="proxy")
    assert proxy["available"] is False
    assert proxy["volume_basis"] == "proxy"


def test_reclaim_when_close_crosses_above_anchor() -> None:
    df = flat_df([100.0, 100.0, 100.0, 110.0])  # vwap 102.5, last close 110
    out = compute_vwap(df)
    assert out["state"] == "reclaimed"


def test_rejection_when_close_crosses_below_anchor() -> None:
    df = flat_df([110.0, 110.0, 110.0, 100.0])  # vwap 107.5, last close 100
    out = compute_vwap(df)
    assert out["state"] == "rejected"


def test_state_above_below_without_fresh_cross() -> None:
    assert compute_vwap(flat_df([90.0, 100.0, 101.0]))["state"] == "above"
    assert compute_vwap(flat_df([110.0, 100.0, 95.0]))["state"] == "below"


def test_trend_slope_classification() -> None:
    # 30min bars: 25 bars stay inside one UTC day (the day anchor resets).
    rising = compute_vwap(flat_df([100.0 + i for i in range(25)], freq="30min"))
    falling = compute_vwap(flat_df([124.0 - i for i in range(25)], freq="30min"))
    flat = compute_vwap(flat_df([100.0] * 25, freq="30min"))
    assert rising["trend"] == "rising"
    assert falling["trend"] == "falling"
    assert flat["trend"] == "flat"


def test_session_window_respects_session_start() -> None:
    idx = pd.date_range(start=DAY, periods=48, freq="1h", tz="UTC")
    close = pd.Series([100.0 + i for i in range(48)], dtype=float, index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 100.0}
    )
    session_start = datetime(2026, 1, 6, 12, 0, tzinfo=timezone.utc)
    out = compute_vwap(df, session_start=session_start)
    # Day anchor resets at UTC midnight: Jan 6 bars are closes 124..147.
    assert out["daily_vwap"] == pytest.approx(135.5)
    # Session anchor: only bars at/after 12:00 -> closes 136..147.
    assert out["session_vwap"] == pytest.approx(141.5)
    assert out["distance_to_session_pct"] == pytest.approx(3.887, abs=0.001)


def test_proxy_volume_untrusted_marks_unavailable() -> None:
    out = compute_vwap(flat_df([100.0, 101.0]), trust_volume=False)
    assert out["available"] is False
    assert "proxy" in out["reason"]
    assert out["daily_vwap"] is None
    assert out["session_vwap"] is None
    assert out["state"] is None
    assert out["trend"] is None


def test_zero_volume_marks_unavailable() -> None:
    idx = pd.date_range(start=DAY, periods=4, freq="1h", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 0.0}
    )
    out = compute_vwap(df)
    assert out["available"] is False
    assert out["reason"] == "no volume data"
    assert out["daily_vwap"] is None
