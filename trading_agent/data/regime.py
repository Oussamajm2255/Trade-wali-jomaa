"""Deterministic regime engine (spec §6).

Six regimes: TREND_UP / TREND_DOWN / RANGE / HIGH_VOLATILITY /
LOW_VOLATILITY / TRANSITION — classified from ADX, ATR behaviour,
EMA alignment and EMA slope. Computed locally, never by the LLM;
DeepSeek only interprets the result. All thresholds are configurable
(regime_* in config.py).
"""
from __future__ import annotations

import pandas as pd

from trading_agent.data.indicators import adx, atr, ema

# A "transition" needs real separation between the moving averages —
# below this (relative to price) we treat alignment as effectively equal.
_TRANSITION_MIN_SEP = 0.0005


def detect_regime(
    df: pd.DataFrame,
    adx_trend: float = 25.0,
    atr_high_mult: float = 1.8,
    atr_low_mult: float = 0.55,
    slope_lookback: int = 5,
    vol_lookback: int = 100,
) -> dict:
    """Classify the market regime from the latest candle's deterministic features."""
    close = df["close"]
    last_close = float(close.iloc[-1])
    adx_now = float(adx(df).iloc[-1])
    atr_now = float(atr(df).iloc[-1])
    atr_median = float(atr(df).dropna().tail(vol_lookback).median()) if len(df) > 1 else atr_now

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200)
    v20, v50, v200 = float(e20.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1])
    slope = float(e20.iloc[-1] - e20.iloc[-slope_lookback])

    # Volatility expansion/contraction relative to the recent ATR median.
    if atr_median > 0 and atr_now >= atr_high_mult * atr_median:
        regime = "high_volatility"
    elif atr_median > 0 and atr_now <= atr_low_mult * atr_median:
        regime = "low_volatility"
    elif v20 > v50 > v200 and slope > 0 and _ema_separated(v20, v50, v200, last_close):
        regime = "trend_up"
    elif v20 < v50 < v200 and slope < 0 and _ema_separated(v20, v50, v200, last_close):
        regime = "trend_down"
    elif _mixed_alignment(v20, v50, v200, last_close):
        regime = "transition"
    elif adx_now >= adx_trend and v20 > v50 and abs(v20 - v50) > _TRANSITION_MIN_SEP * last_close:
        regime = "trend_up"
    elif adx_now >= adx_trend and v20 < v50 and abs(v20 - v50) > _TRANSITION_MIN_SEP * last_close:
        regime = "trend_down"
    else:
        regime = "range"

    return {
        "regime": regime,
        "adx": round(adx_now, 2),
        "atr": round(atr_now, 8),
        "atr_ratio_vs_median": round(atr_now / atr_median, 4) if atr_median > 0 else None,
        "ema_alignment": "bull" if v20 > v50 > v200 else "bear" if v20 < v50 < v200 else "mixed",
        "ema20_slope": round(slope, 8),
        "trend_strength": round(min(1.0, adx_now / 50.0), 4),
    }


def _mixed_alignment(v20: float, v50: float, v200: float, price: float) -> bool:
    """EMA20 and EMA50 point in different directions vs EMA200, with real
    separation (float-noise is not a transition)."""
    sep = _TRANSITION_MIN_SEP * price
    return (v20 > v50) != (v50 > v200) and abs(v20 - v50) > sep and abs(v50 - v200) > sep


def _ema_separated(v20: float, v50: float, v200: float, price: float) -> bool:
    """True when the EMA20/50/200 chain has real separation at every step.

    Without this, tiny float differences (flat noise) would be labelled
    trend_up/trend_down — a false trend from nothing."""
    sep = _TRANSITION_MIN_SEP * price
    return abs(v20 - v50) > sep and abs(v50 - v200) > sep
