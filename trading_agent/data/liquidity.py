"""Liquidity map (V-MONSTER §9): named levels, distances, quality.

Built from the canonical snapshot's own candles + structure events:
previous day/week/month highs and lows, today's Asia/London/New York
session ranges, swing highs/lows, equal highs/lows. Everything is
deterministic — unavailable levels are simply absent (never fabricated,
spec §4).
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from trading_agent.data.indicators import atr

# Standard intraday session ranges in LOCAL time (summer/winter offsets
# are applied automatically by ZoneInfo).
ASIA_TZ = ZoneInfo("Asia/Tokyo")
LONDON_TZ = ZoneInfo("Europe/London")
NY_TZ = ZoneInfo("America/New_York")

DEFAULT_ASIA = "09:00-18:00"      # Tokyo local -> 00:00-09:00 UTC
DEFAULT_LONDON = "08:00-17:00"    # London local
DEFAULT_NEW_YORK = "09:30-17:00"  # New York local

_LEVEL_TOLERANCE = 0.0002  # levels within 0.02% of price are "touched", not a reference


def _parse_window(window: str) -> tuple[int, int] | None:
    try:
        start, end = window.split("-")
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
        return sh * 60 + sm, eh * 60 + em
    except (ValueError, AttributeError):
        return None


def _session_high_low(
    df: pd.DataFrame, now: datetime, tz: ZoneInfo, window: str
) -> tuple[float | None, float | None]:
    """High/low of today's candles inside one session window (local)."""
    parsed = _parse_window(window)
    if parsed is None:
        return None, None
    start_min, end_min = parsed
    local_day = now.astimezone(tz).date()
    start_ts = pd.Timestamp(
        datetime.combine(local_day, datetime.min.time(), tzinfo=tz)
    ) + pd.Timedelta(minutes=start_min)
    end_ts = pd.Timestamp(
        datetime.combine(local_day, datetime.min.time(), tzinfo=tz)
    ) + pd.Timedelta(minutes=end_min)
    window_df = df[(df.index >= start_ts.astimezone("UTC")) & (df.index <= end_ts.astimezone("UTC"))]
    if window_df.empty:
        return None, None
    return round(float(window_df["high"].max()), 8), round(float(window_df["low"].min()), 8)


def _previous_month_high_low(daily_df: pd.DataFrame | None, now: pd.Timestamp) -> tuple[float | None, float | None]:
    """High/low of the last fully closed calendar month before this one."""
    if daily_df is None or daily_df.empty:
        return None, None
    first_of_month = now.normalize().replace(day=1)
    prev = daily_df[daily_df.index < first_of_month]
    if prev.empty:
        return None, None
    last_month_end = first_of_month - pd.Timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    month = prev[(prev.index >= last_month_start) & (prev.index <= last_month_end)]
    if month.empty:
        month = prev.tail(21)
    return round(float(month["high"].max()), 8), round(float(month["low"].min()), 8)


def compute_liquidity(
    entry_df: pd.DataFrame,
    gold_context: dict,
    structure: dict,
    now: datetime | None = None,
    daily_df: pd.DataFrame | None = None,
    asia: str = DEFAULT_ASIA,
    london: str = DEFAULT_LONDON,
    new_york: str = DEFAULT_NEW_YORK,
) -> dict:
    """Assemble the liquidity map for one cycle (V-MONSTER §9).

    Named levels come from three families:
    - time levels: previous day/week/month highs/lows + today's session
      ranges (Asia/London/NY);
    - structure levels: swing highs/lows and equal highs/lows from the
      structure engine;
    - price distance: nearest level above/below and LIQUIDITY_QUALITY 0-1.

    `daily_df` (1d candles) enables the previous-month levels (PMH/PML);
    without it they are simply absent — never fabricated (spec §4).
    """
    now = now or datetime.now().astimezone()
    price = float(entry_df["close"].iloc[-1])
    atr_now = float(atr(entry_df).iloc[-1]) if len(entry_df) >= 14 else None

    levels: list[dict] = []

    def add(price_value: float | None, kind: str) -> None:
        if price_value is None:
            return
        price_value = round(float(price_value), 8)
        if any(abs(price_value - lv["price"]) < 1e-8 for lv in levels):
            return
        levels.append({"price": price_value, "kind": kind})

    add(gold_context.get("prev_day_high"), "PDH")
    add(gold_context.get("prev_day_low"), "PDL")
    add(gold_context.get("prev_week_high"), "PWH")
    add(gold_context.get("prev_week_low"), "PWL")
    add(gold_context.get("intraday_high"), "INTRADAY_HIGH")
    add(gold_context.get("intraday_low"), "INTRADAY_LOW")
    add(gold_context.get("session_high"), "SESSION_HIGH")
    add(gold_context.get("session_low"), "SESSION_LOW")

    now_ts = pd.Timestamp(now)
    pmh, pml = _previous_month_high_low(daily_df, now_ts)
    add(pmh, "PMH")
    add(pml, "PML")

    asia_high, asia_low = _session_high_low(entry_df, now, ASIA_TZ, asia)
    london_high, london_low = _session_high_low(entry_df, now, LONDON_TZ, london)
    ny_high, ny_low = _session_high_low(entry_df, now, NY_TZ, new_york)
    add(asia_high, "ASIA_HIGH")
    add(asia_low, "ASIA_LOW")
    add(london_high, "LONDON_HIGH")
    add(london_low, "LONDON_LOW")
    add(ny_high, "NY_HIGH")
    add(ny_low, "NY_LOW")

    for swing in structure.get("swings", []) or []:
        kind = "SWING_HIGH" if swing.get("type") == "SWING_HIGH" else "SWING_LOW"
        add(swing.get("price"), kind)
    for eq in structure.get("equal_highs", []) or []:
        add(eq, "EQH")
    for eq in structure.get("equal_lows", []) or []:
        add(eq, "EQL")

    above = [lv for lv in levels if lv["price"] > price * (1 + _LEVEL_TOLERANCE)]
    below = [lv for lv in levels if lv["price"] < price * (1 - _LEVEL_TOLERANCE)]
    above.sort(key=lambda lv: lv["price"])
    below.sort(key=lambda lv: lv["price"], reverse=True)

    def nearest(side: list[dict]) -> dict | None:
        if not side:
            return None
        lv = side[0]
        distance_pct = round((lv["price"] / price - 1) * 100, 3)
        out = {"price": lv["price"], "kind": lv["kind"], "distance_pct": distance_pct}
        if atr_now:
            out["distance_atr"] = round(abs(lv["price"] - price) / atr_now, 2)
        return out

    nearest_below = nearest(below)
    nearest_above = nearest(above)

    # LIQUIDITY_QUALITY 0-1: proximity of both sides (0.5), level density
    # (0.3), freshness of the time levels (0.2). Explainable and bounded.
    def proximity(level: dict | None) -> float:
        if level is None or atr_now is None:
            return 0.0
        dist_atr = abs(level["price"] - price) / atr_now
        if 0.3 <= dist_atr <= 4:
            return 1.0
        if 0.3 <= dist_atr <= 8:
            return 0.5
        return 0.2

    side_score = (proximity(nearest_below) + proximity(nearest_above)) / 2
    density = (min(len(above), 3) + min(len(below), 3)) / 6
    time_levels = [
        gold_context.get(k) for k in ("prev_day_high", "prev_day_low", "prev_week_high", "prev_week_low")
    ]
    freshness = sum(1 for v in time_levels if v is not None) / 4
    quality = round(0.5 * side_score + 0.3 * density + 0.2 * freshness, 2)

    return {
        "price": round(price, 8),
        "atr": atr_now,
        "levels": levels[:16],
        "nearest_below": nearest_below,
        "nearest_above": nearest_above,
        "quality": quality,
    }
