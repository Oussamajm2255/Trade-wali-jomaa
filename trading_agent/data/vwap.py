"""Session & daily VWAP (V-MONSTER §12): volume-weighted anchors.

Built from the canonical snapshot's own candles (typical price ×
volume). Honest about data (spec §4): when volume is untrusted (proxy
fallback — token flow, not gold flow) or absent/zero, the VWAP is
marked unavailable with the reason instead of a misleading number.
The reclaim/rejection state is a fresh cross of the close against the
anchor; the trend is the slope of the cumulative VWAP over the last
bars.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

_VWAP_TREND_BARS = 20  # slope measured over the last N bars
_VWAP_TREND_THRESHOLD_PCT = 0.05  # 0.05% move -> rising/falling


def _has_volume(df: pd.DataFrame) -> bool:
    if df.empty or "volume" not in df.columns:
        return False
    return bool(float(df["volume"].fillna(0.0).sum()) > 0)


def _vwap_series(window: pd.DataFrame) -> pd.Series:
    """Cumulative typical-price VWAP over `window` (NaN while volume is 0)."""
    tp = (window["high"] + window["low"] + window["close"]) / 3
    vol = window["volume"].fillna(0.0)
    cum_pv = (tp * vol).cumsum()
    cum_v = vol.cumsum()
    return cum_pv / cum_v.replace(0.0, pd.NA)


def _vwap_value(window: pd.DataFrame) -> float | None:
    if window.empty:
        return None
    series = _vwap_series(window)
    if pd.isna(series.iloc[-1]):
        return None
    return round(float(series.iloc[-1]), 8)


def _trend(window: pd.DataFrame) -> str | None:
    """VWAP slope over the last `_VWAP_TREND_BARS` bars of the window."""
    if len(window) < 2:
        return None
    series = _vwap_series(window)
    last = series.iloc[-1]
    if pd.isna(last):
        return None
    ref = series.iloc[max(0, len(window) - 1 - _VWAP_TREND_BARS)]
    if pd.isna(ref) or float(ref) == 0:
        return None
    change_pct = (float(last) / float(ref) - 1) * 100
    if change_pct > _VWAP_TREND_THRESHOLD_PCT:
        return "rising"
    if change_pct < -_VWAP_TREND_THRESHOLD_PCT:
        return "falling"
    return "flat"


def _cross_state(window: pd.DataFrame, vwap: float | None) -> str | None:
    """Close vs anchor: fresh reclaim/reject, otherwise above/below."""
    if vwap is None or window.empty:
        return None
    last_close = float(window["close"].iloc[-1])
    if len(window) == 1:
        return "above" if last_close > vwap else "below"
    prev_close = float(window["close"].iloc[-2])
    last_side = "above" if last_close > vwap else "below"
    prev_side = "above" if prev_close > vwap else "below"
    if last_side == "above" and prev_side == "below":
        return "reclaimed"
    if last_side == "below" and prev_side == "above":
        return "rejected"
    return last_side


def _pct(price: float, vwap: float | None) -> float | None:
    if vwap is None:
        return None
    return round((price / vwap - 1) * 100, 3)


def compute_vwap(
    entry_df: pd.DataFrame,
    session_start: datetime | None = None,
    trust_volume: bool = True,
) -> dict:
    """Session + daily VWAP for one cycle (V-MONSTER §12).

    - `daily_vwap`: volume-weighted anchor over today's candles (UTC day).
    - `session_vwap`: same, restricted to the active session window
      (`session_start` UTC); None while the session has no candles yet.
    - `state`: fresh close cross against the session anchor (fallback:
      day anchor) — reclaimed / rejected / above / below.
    - `trend`: slope of the day-window cumulative VWAP.
    """
    if not trust_volume:
        return {
            "available": False,
            "reason": "volume untrusted (proxy fallback)",
            "daily_vwap": None,
            "session_vwap": None,
            "distance_to_daily_pct": None,
            "distance_to_session_pct": None,
            "state": None,
            "trend": None,
        }
    if not _has_volume(entry_df):
        return {
            "available": False,
            "reason": "no volume data",
            "daily_vwap": None,
            "session_vwap": None,
            "distance_to_daily_pct": None,
            "distance_to_session_pct": None,
            "state": None,
            "trend": None,
        }

    price = float(entry_df["close"].iloc[-1])
    day_start = pd.Timestamp(entry_df.index[-1]).normalize()
    day_window = entry_df[entry_df.index >= day_start]
    if day_window.empty:
        day_window = entry_df

    session_window = day_window
    if session_start is not None:
        start_ts = pd.Timestamp(session_start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        session_window = entry_df[entry_df.index >= start_ts]
        if session_window.empty:
            session_window = day_window

    daily_vwap = _vwap_value(day_window)
    session_vwap = _vwap_value(session_window)

    anchor_window = session_window if session_vwap is not None else day_window
    anchor = session_vwap if session_vwap is not None else daily_vwap

    return {
        "available": True,
        "reason": None,
        "daily_vwap": daily_vwap,
        "session_vwap": session_vwap,
        "distance_to_daily_pct": _pct(price, daily_vwap),
        "distance_to_session_pct": _pct(price, session_vwap),
        "state": _cross_state(anchor_window, anchor),
        "trend": _trend(day_window),
    }
