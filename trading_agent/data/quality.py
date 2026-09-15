"""Data-quality validation (spec §4): PASS / DEGRADED / FAIL.

- FAIL     -> stop analysis: no AI calls, no proposal.
- DEGRADED -> continue only if configuration allows; the state and its
  reasons are labelled and stored with every signal / rejection.
- PASS     -> normal analysis.

Every check produces a named issue; the overall state is the worst
severity found. Nothing here fetches data — it only inspects what the
canonical snapshot builder already gathered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

# Candle duration per timeframe string (for staleness checks).
CANDLE_DURATION = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

_SEVERITY = {"pass": 0, "degraded": 1, "fail": 2}


class QualityState(StrEnum):
    PASS = "pass"
    DEGRADED = "degraded"
    FAIL = "fail"


def _worst(a: QualityState, b: QualityState) -> QualityState:
    return a if _SEVERITY[a.value] >= _SEVERITY[b.value] else b


@dataclass
class DataQuality:
    """Aggregate validation result for one data source or a whole cycle."""

    state: QualityState = QualityState.PASS
    issues: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)  # check name -> state

    def add(self, check: str, state: QualityState, issue: str) -> "DataQuality":
        self.checks[check] = state.value
        self.state = _worst(self.state, state)
        if state != QualityState.PASS:
            self.issues.append(f"[{check}] {issue}")
        return self


def validate_candles(
    df: pd.DataFrame,
    timeframe: str,
    min_candles: int = 60,
    max_stale_multiple: float = 3.0,
) -> DataQuality:
    """Validate one OHLCV frame: ordering, duplicates, OHLC sanity, history, freshness."""
    q = DataQuality()
    if df is None or len(df) == 0:
        return q.add("empty", QualityState.FAIL, "no candles")
    if not df.index.is_monotonic_increasing:
        q.add("ordering", QualityState.FAIL, "timestamps not ascending")
    dupes = int(df.index.duplicated().sum())
    if dupes:
        q.add("duplicates", QualityState.DEGRADED, f"{dupes} duplicated timestamp(s)")
    ohlc = df[["open", "high", "low", "close"]]
    if ohlc.isna().any().any():
        q.add("nan", QualityState.FAIL, "NaN in OHLC")
    if (ohlc <= 0).any().any():
        q.add("invalid_ohlc", QualityState.FAIL, "zero/negative OHLC values")
    if (df["high"] < df["low"]).any():
        q.add("invalid_ohlc", QualityState.FAIL, "high < low")
    if len(df) < min_candles:
        q.add("min_candles", QualityState.FAIL, f"only {len(df)} candles (< {min_candles})")
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    age = pd.Timestamp.now(tz="UTC") - last_ts
    if age > max_stale_multiple * pd.Timedelta(CANDLE_DURATION.get(timeframe, "1h")):
        q.add(
            "stale",
            QualityState.DEGRADED,
            f"last candle is {age} old (> {max_stale_multiple}x {timeframe})",
        )
    return q


def validate_gauge(gauge: dict | None, max_age_hours: int = 48) -> DataQuality:
    """DXY-gauge sanity: presence (for the DXY contract) and freshness."""
    q = DataQuality()
    if not gauge:
        return q.add("dxy_unavailable", QualityState.DEGRADED, "DXY gauge unavailable")
    if gauge.get("kind") != "dxy":
        return q  # e.g. crypto Fear & Greed: different contract, no check
    try:
        ts = pd.Timestamp(gauge["ts"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age = pd.Timestamp.now(tz="UTC") - ts
    except (KeyError, ValueError):
        return q.add("dxy_ts", QualityState.DEGRADED, "DXY gauge timestamp unreadable")
    if age > pd.Timedelta(hours=max_age_hours):
        q.add("dxy_stale", QualityState.DEGRADED, f"DXY gauge is {age} old (> {max_age_hours}h)")
    return q


def validate_indicators(snapshot: dict) -> DataQuality:
    """Indicator sanity on a computed snapshot (invalid ATR cannot be sized)."""
    q = DataQuality()
    atr = snapshot.get("atr_14")
    if atr is None or pd.isna(atr) or atr <= 0:
        q.add("invalid_atr", QualityState.FAIL, f"ATR is {atr!r} — cannot compute a valid stop")
    return q


def combine_quality(*items: DataQuality | None) -> DataQuality:
    """Merge per-source results: worst state wins, issues concatenated."""
    out = DataQuality()
    for item in items:
        if item is None:
            continue
        out.state = _worst(out.state, item.state)
        out.issues.extend(item.issues)
        out.checks.update(item.checks)
    return out
