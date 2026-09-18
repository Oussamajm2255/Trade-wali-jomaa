"""Counterfactual entry timing (V-MONSTER §66, research-only).

"What if the human had entered T±3/5/10s around the signal?" The only
intra-bar data available is the M1 OHLCV series, so the counterfactual
entry price is a linear open→close interpolation inside the M1 bar that
contains the shifted timestamp. That is a documented approximation
(1-minute resolution), never presented as a tick-accurate fill — the
label carries it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_agent.schema.types import Side

# The spec's research grid: seconds before/after the signal timestamp.
CF_OFFSETS_S = (-10, -5, -3, 0, 3, 5, 10)

METHOD = "linear_interpolation_m1"


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def counterfactual_entry(
    m1_candles: list[dict],
    base_ts: datetime,
    offset_s: int,
) -> dict | None:
    """Estimated entry price at base_ts + offset_s from M1 bars.

    `m1_candles` is a sorted list of {"ts": datetime, "open": f,
    "close": f}. None when no bar covers the shifted timestamp (data
    gap) — the caller must label the result honestly, never invent.
    """
    target = _as_utc(base_ts) + timedelta(seconds=offset_s)
    for bar in m1_candles:
        bar_ts = bar.get("ts")
        if bar_ts is None:
            continue
        bar_start = _as_utc(bar_ts)
        bar_end = bar_start + timedelta(minutes=1)
        if bar_start <= target < bar_end:
            fraction = (target - bar_start).total_seconds() / 60.0
            o, c = float(bar["open"]), float(bar["close"])
            price = o + (c - o) * fraction
            return {
                "offset_s": offset_s,
                "ts": target,
                "price": round(price, 8),
                "bar_index": m1_candles.index(bar),
                "method": METHOD,
            }
    return None


def counterfactual_rr(
    side: Side,
    entry: float,
    stop: float,
    target: float,
    cf_entry: float,
) -> float | None:
    """The what-if R:R of entering at the counterfactual price instead.

    Stop and target stay as planned (they are plan-based, not price-
    based); only the entry shifts. None when the shifted entry is
    beyond the stop (the counterfactual entry would be nonsense).
    """
    if side == Side.LONG:
        if cf_entry <= stop:
            return None
        return round((target - cf_entry) / (cf_entry - stop), 4)
    if side == Side.SHORT:
        if cf_entry >= stop:
            return None
        return round((cf_entry - target) / (stop - cf_entry), 4)
    return None


def counterfactual_report(
    m1_candles: list[dict],
    base_ts: datetime,
    side: Side,
    entry: float,
    stop: float,
    target: float,
    offsets_s: tuple[int, ...] = CF_OFFSETS_S,
) -> list[dict]:
    """One research row per offset: estimated entry, drift, what-if R:R.

    Missing M1 coverage yields {"offset_s", "available": False} — the
    data honesty rule applies to research too.
    """
    report: list[dict] = []
    for offset in offsets_s:
        cf = counterfactual_entry(m1_candles, base_ts, offset)
        if cf is None:
            report.append({"offset_s": offset, "available": False})
            continue
        cf["available"] = True
        cf["drift"] = round(cf["price"] - entry, 8)
        cf["rr"] = counterfactual_rr(side, entry, stop, target, cf["price"])
        report.append(cf)
    return report
