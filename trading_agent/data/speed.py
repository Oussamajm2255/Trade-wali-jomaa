"""Market speed (V-MONSTER §27) — deterministic, candle-based.

How fast the market is moving right now, measured on three honest axes
computable from OHLCV alone:

  - range per minute vs ATR per minute (sustained candle range),
  - candle formation speed (how much of its typical range the current
    candle has already consumed, normalized by elapsed candle time),
  - volatility acceleration (ATR now vs ATR `accel_lookback` candles
    ago).

Classification: SLOW / NORMAL / FAST / EXTREME. Insufficient history
fails open to NORMAL (a young market is unknown, not fast) — the same
principle as the shock engine. The function is pure: same candles ->
same verdict, so backtests stay reproducible.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from trading_agent.data.indicators import atr as atr_series


class SpeedState:
    SLOW = "SLOW"
    NORMAL = "NORMAL"
    FAST = "FAST"
    EXTREME = "EXTREME"


_TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "1d": 1440,
}


def _timeframe_minutes(timeframe: str) -> int:
    """Resolve a timeframe label to minutes; unknown labels default to 60."""
    if timeframe in _TIMEFRAME_MINUTES:
        return _TIMEFRAME_MINUTES[timeframe]
    digits = "".join(ch for ch in timeframe if ch.isdigit())
    if digits:
        value = int(digits)
        if timeframe.endswith("h"):
            return value * 60
        if timeframe.endswith("d"):
            return value * 1440
        return value
    return 60


def compute_market_speed(
    df: pd.DataFrame,
    timeframe: str = "15m",
    *,
    window: int = 12,
    fast_mult: float = 1.8,
    extreme_mult: float = 3.0,
    slow_mult: float = 0.4,
    accel_lookback: int = 12,
    accel_extreme_mult: float = 1.5,
    now: datetime | None = None,
) -> dict:
    """Classify the market speed of the last candle of `df`.

    `now` anchors the elapsed fraction of the current candle: live
    cycles pass wall-clock time (mid-candle), historical replays pass
    the replay timestamp. When the candle is closed (or `now` is None)
    the formation ratio reduces to the candle's range vs the typical
    range.

    Returns {"state", "range_per_minute", "atr_per_minute",
    "range_ratio", "formation_ratio", "acceleration", "detail",
    "insufficient_history"}.
    """
    out = {
        "state": SpeedState.NORMAL,
        "range_per_minute": None,
        "atr_per_minute": None,
        "range_ratio": None,
        "formation_ratio": None,
        "acceleration": None,
        "detail": "",
        "insufficient_history": False,
    }
    if len(df) < max(window, accel_lookback) + 1:
        out["detail"] = "insufficient history for speed baseline"
        out["insufficient_history"] = True
        return out

    minutes = _timeframe_minutes(timeframe)
    ranges = (df["high"] - df["low"]).astype(float)
    atr_vals = atr_series(df)
    atr_now = float(atr_vals.iloc[-1])
    mean_range = float(ranges.iloc[-window - 1 : -1].mean())
    if atr_now <= 0 or pd.isna(atr_now) or mean_range <= 0 or pd.isna(mean_range):
        out["detail"] = "no ATR baseline"
        out["insufficient_history"] = True
        return out

    out["range_per_minute"] = round(mean_range / minutes, 6)
    out["atr_per_minute"] = round(atr_now / minutes, 6)
    out["range_ratio"] = round(mean_range / atr_now, 4)

    # Formation speed: typical-range fraction consumed, normalized by
    # elapsed candle time. A closed candle (elapsed = full duration) is
    # simply its range vs the typical range.
    cur_range = float(ranges.iloc[-1])
    cur_time = df.index[-1]
    if now is not None:
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None and cur_time.tzinfo is not None:
            now_ts = now_ts.tz_localize("UTC")
        elif now_ts.tzinfo is not None and cur_time.tzinfo is None:
            now_ts = now_ts.tz_localize(None)
        elapsed = min(
            float(minutes), max(0.0, (now_ts - cur_time).total_seconds() / 60.0)
        )
    else:
        elapsed = float(minutes)
    fraction = max(0.5, elapsed / minutes)  # floor: a fresh candle's
    # first ticks are noise, not speed — a brand-new candle can never
    # score more than 2x its closed-candle formation ratio.
    out["formation_ratio"] = round((cur_range / mean_range) / fraction, 4)

    # Volatility acceleration: ATR now vs ATR accel_lookback candles ago.
    atr_lag = float(atr_vals.iloc[-accel_lookback - 1])
    accel = atr_now / atr_lag if atr_lag > 0 and not pd.isna(atr_lag) else 1.0
    out["acceleration"] = round(accel, 4)

    base = max(out["range_ratio"], out["formation_ratio"])
    if base >= extreme_mult or (base >= fast_mult and accel >= accel_extreme_mult):
        state = SpeedState.EXTREME
    elif base >= fast_mult:
        state = SpeedState.FAST
    elif out["formation_ratio"] <= slow_mult and accel <= 1.0:
        # SLOW keys on the candle's own pace: the current candle has
        # consumed less than slow_mult of its typical range while ATR is
        # not expanding (genuinely quiet, not the calm before a surge).
        # range_ratio measures baseline consistency, not pace — it must
        # not gate SLOW (it stays ~1.0 for any self-consistent market).
        state = SpeedState.SLOW
    else:
        state = SpeedState.NORMAL
    out["state"] = state
    out["detail"] = (
        f"{state.lower()}: range {out['range_ratio']:.2f}x ATR, formation "
        f"{out['formation_ratio']:.2f}x, acceleration {accel:.2f}x"
    )
    return out
