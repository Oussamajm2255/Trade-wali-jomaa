"""Gold-specific context (spec §10).

Key levels and distances computed from the canonical snapshot's own
candles: daily/weekly opens, previous day/week highs and lows, session
high/low and the volatility percentile. Everything is deterministic —
DeepSeek reads these numbers, it does not derive them.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from trading_agent.data.indicators import atr_percentile


def _pct_distance(price: float, level: float | None) -> float | None:
    if level is None or not level:
        return None
    return round((price / level - 1) * 100, 3)


def _first_candle_open_since(df: pd.DataFrame, since: pd.Timestamp) -> float | None:
    """Open of the first candle at/after `since` (daily/weekly open)."""
    window = df[df.index >= since]
    if window.empty:
        return None
    return round(float(window["open"].iloc[0]), 8)


def _intraday_high_low(df: pd.DataFrame, since: pd.Timestamp) -> tuple[float | None, float | None]:
    window = df[df.index >= since]
    if window.empty:
        return None, None
    return round(float(window["high"].max()), 8), round(float(window["low"].min()), 8)


def _previous_week_high_low(daily_df: pd.DataFrame | None, now: pd.Timestamp) -> tuple[float | None, float | None]:
    """High/low of the last fully closed week (Mon-Fri) before this week."""
    if daily_df is None or daily_df.empty:
        return None, None
    this_monday = (now - pd.Timedelta(days=now.weekday())).normalize()
    prev = daily_df[daily_df.index < this_monday]
    if prev.empty:
        return None, None
    last_week_end = this_monday - pd.Timedelta(days=1)
    last_week_start = last_week_end - pd.Timedelta(days=6)
    week = prev[(prev.index >= last_week_start) & (prev.index <= last_week_end)]
    if week.empty:
        week = prev.tail(5)
    return round(float(week["high"].max()), 8), round(float(week["low"].min()), 8)


def compute_gold_context(
    entry_df: pd.DataFrame,
    daily_df: pd.DataFrame | None = None,
    session_start: datetime | None = None,
) -> dict:
    """Key levels + distances for one analysis cycle (spec §10).

    `entry_df` are the entry-timeframe candles; `daily_df` the 1d candles
    (optional — previous day/week levels are None without it).
    `session_start` is the UTC start of the active session (from
    sessions.session_start_utc); session high/low fall back to the UTC
    intraday range when no session is running.
    """
    price = float(entry_df["close"].iloc[-1])
    now = pd.Timestamp.now(tz="UTC")

    day_start = now.normalize()
    week_start = (now - pd.Timedelta(days=now.weekday())).normalize()
    daily_open = _first_candle_open_since(entry_df, day_start)
    weekly_open = _first_candle_open_since(entry_df, week_start)

    pdh = pdl = pwh = pwl = None
    if daily_df is not None and not daily_df.empty:
        last_closed_day = daily_df.iloc[-1]
        pdh, pdl = round(float(last_closed_day["high"]), 8), round(float(last_closed_day["low"]), 8)
    pwh, pwl = _previous_week_high_low(daily_df, now)

    session_start_ts = pd.Timestamp(session_start) if session_start is not None else day_start
    session_high, session_low = _intraday_high_low(entry_df, session_start_ts)
    intraday_high, intraday_low = _intraday_high_low(entry_df, day_start)

    return {
        "price": round(price, 8),
        "daily_open": daily_open,
        "weekly_open": weekly_open,
        "prev_day_high": pdh,
        "prev_day_low": pdl,
        "prev_week_high": pwh,
        "prev_week_low": pwl,
        "session_high": session_high,
        "session_low": session_low,
        "intraday_high": intraday_high,
        "intraday_low": intraday_low,
        "distance_from_daily_open_pct": _pct_distance(price, daily_open),
        "distance_from_pdh_pct": _pct_distance(price, pdh),
        "distance_from_pdl_pct": _pct_distance(price, pdl),
        "volatility_percentile_100": round(atr_percentile(entry_df), 2),
    }
