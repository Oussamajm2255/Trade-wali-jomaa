"""Liquidity map (V-MONSTER §9): named levels, distances, quality."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from trading_agent.data.liquidity import (
    ASIA_TZ,
    _parse_window,
    _previous_month_high_low,
    _session_high_low,
    compute_liquidity,
)

# A fixed "now" two days AFTER the synthetic candles end: session windows
# on that day contain no bars, so levels are fully deterministic.
END = pd.Timestamp("2026-09-17 12:00", tz="UTC")
NOW = datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc)


def make_df(n: int = 300, base: float = 2490.0, step: float = 0.05) -> pd.DataFrame:
    idx = pd.date_range(end=END, periods=n, freq="15min", tz="UTC")
    close = base + pd.Series(range(n), dtype=float).to_numpy() * step
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 100.0},
        index=idx,
    )


def named_gold() -> dict:
    return {
        "prev_day_high": 2500.0,
        "prev_day_low": 2480.0,
        "prev_week_high": 2520.0,
        "prev_week_low": 2470.0,
        "intraday_high": 2510.0,
        "intraday_low": 2490.0,
        "session_high": 2505.0,
        "session_low": 2495.0,
    }


def kinds(liq: dict) -> dict[str, float]:
    return {lv["kind"]: lv["price"] for lv in liq["levels"]}


def test_named_time_levels_carried() -> None:
    liq = compute_liquidity(make_df(), named_gold(), {}, now=NOW)
    found = kinds(liq)
    assert found["PDH"] == 2500.0
    assert found["PDL"] == 2480.0
    assert found["PWH"] == 2520.0
    assert found["PWL"] == 2470.0
    assert found["INTRADAY_HIGH"] == 2510.0
    assert found["SESSION_LOW"] == 2495.0


def test_nearest_levels_and_distance_math() -> None:
    liq = compute_liquidity(make_df(), named_gold(), {}, now=NOW)
    # Session high (2505) is within the 0.02% touch tolerance of price
    # (2504.95) -> not a reference; nearest above is the intraday high.
    above = liq["nearest_above"]
    below = liq["nearest_below"]
    assert above["kind"] == "INTRADAY_HIGH"
    assert above["price"] == 2510.0
    assert above["distance_pct"] == pytest.approx(0.202, abs=0.001)
    assert above["distance_atr"] == pytest.approx(8.42, abs=0.05)
    assert below["kind"] == "PDH"
    assert below["price"] == 2500.0
    assert below["distance_pct"] == pytest.approx(-0.198, abs=0.001)
    assert below["distance_atr"] == pytest.approx(8.25, abs=0.05)
    # Sanity: distances signed correctly around price.
    assert above["price"] > liq["price"] > below["price"]


def test_swings_and_equal_levels_added() -> None:
    structure = {
        "swings": [
            {"type": "SWING_HIGH", "price": 2512.0},
            {"type": "SWING_LOW", "price": 2488.0},
        ],
        "equal_highs": [2503.0],
        "equal_lows": [2493.0],
    }
    liq = compute_liquidity(make_df(), named_gold(), structure, now=NOW)
    found = kinds(liq)
    assert found["SWING_HIGH"] == 2512.0
    assert found["SWING_LOW"] == 2488.0
    assert found["EQH"] == 2503.0
    assert found["EQL"] == 2493.0


def test_duplicate_prices_merge_into_one_level() -> None:
    structure = {"swings": [{"type": "SWING_HIGH", "price": 2500.0}]}  # == PDH
    liq = compute_liquidity(make_df(), named_gold(), structure, now=NOW)
    assert sum(1 for lv in liq["levels"] if lv["price"] == 2500.0) == 1


def test_quality_bounded_and_proximity_driven() -> None:
    near = {"prev_day_high": 2506.95, "prev_day_low": 2502.95}  # ~3.3 ATR away
    far = {"prev_day_high": 2514.95, "prev_day_low": 2494.95}  # ~16.7 ATR away
    liq_near = compute_liquidity(make_df(), near, {}, now=NOW)
    liq_far = compute_liquidity(make_df(), far, {}, now=NOW)
    assert 0.0 <= liq_near["quality"] <= 1.0
    assert 0.0 <= liq_far["quality"] <= 1.0
    assert liq_near["quality"] > liq_far["quality"]


def test_distance_atr_absent_when_atr_unavailable() -> None:
    liq = compute_liquidity(make_df(n=10), named_gold(), {}, now=NOW)
    assert liq["atr"] is None
    assert "distance_atr" not in liq["nearest_above"]


def test_previous_month_high_low() -> None:
    idx = pd.date_range(start="2026-07-01", periods=76, freq="1D", tz="UTC")
    high = pd.Series(
        [100.0] * 31 + [110.0] * 31 + [200.0] * 14, index=idx, dtype=float
    )
    low = pd.Series(
        [90.0] * 31 + [95.0] * 31 + [150.0] * 14, index=idx, dtype=float
    )
    df = pd.DataFrame({"open": high, "high": high, "low": low, "close": high, "volume": 1.0})
    pmh, pml = _previous_month_high_low(df, pd.Timestamp("2026-09-14", tz="UTC"))
    assert pmh == 110.0  # August, the last fully closed month
    assert pml == 95.0


def test_monthly_levels_via_daily_df() -> None:
    idx = pd.date_range(start="2026-07-01", periods=76, freq="1D", tz="UTC")
    high = pd.Series([100.0] * 31 + [110.0] * 31 + [200.0] * 14, index=idx, dtype=float)
    low = pd.Series([90.0] * 31 + [95.0] * 31 + [150.0] * 14, index=idx, dtype=float)
    df = pd.DataFrame({"open": high, "high": high, "low": low, "close": high, "volume": 1.0})
    liq = compute_liquidity(
        make_df(), named_gold(), {}, now=datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc), daily_df=df
    )
    found = kinds(liq)
    assert found["PMH"] == 110.0
    assert found["PML"] == 95.0


def test_session_window_high_low() -> None:
    idx = pd.date_range(start="2026-09-14 00:00", periods=48, freq="1h", tz="UTC")
    high = pd.Series(range(48), dtype=float, index=idx)
    df = pd.DataFrame(
        {"open": high, "high": high, "low": high - 10.0, "close": high, "volume": 1.0}
    )
    now = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)  # 21:00 Tokyo, same day
    hi, lo = _session_high_low(df, now, ASIA_TZ, "09:00-18:00")
    # Tokyo 09:00-18:00 == UTC 00:00-09:00 -> bars 0..9.
    assert hi == 9.0
    assert lo == -10.0


def test_parse_window() -> None:
    assert _parse_window("09:00-18:00") == (540, 1080)
    assert _parse_window("") is None
    assert _parse_window("garbage") is None
