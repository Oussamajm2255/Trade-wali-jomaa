"""Deterministic DXY context (spec §9).

The existing gauge stays the compatibility gate (LONG needs gauge >=
dxy_long_min, SHORT needs <= dxy_short_max). This module adds the richer
context around it: level, direction, momentum, 15m/1h/4h changes, trend
and volatility — all computed from real DXY candles, never by the LLM —
plus a 0-100 context score that keeps the existing semantic:

    100 = weak dollar = bullish gold
    0   = strong dollar = bearish gold

When intraday DXY candles are unavailable the context degrades to the
gauge-only interpretation and says so, instead of inventing numbers.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from trading_agent.data.indicators import adx, atr, ema

_DXY_DIRECTION_THRESHOLD_PCT = 0.1  # ROC below this = "flat"


def _pct_change(df: pd.DataFrame, lookback: int) -> float | None:
    """Close-to-close percent change over `lookback` candles, or None."""
    close = df["close"]
    if len(close) <= lookback or float(close.iloc[-lookback - 1]) == 0:
        return None
    return round((float(close.iloc[-1]) / float(close.iloc[-lookback - 1]) - 1) * 100, 3)


def _changes(df: pd.DataFrame) -> dict:
    """Changes for the granularities this frame can honestly support."""
    duration = pd.Timedelta(df.index[-1] - df.index[-2]) if len(df) > 1 else None
    changes: dict = {}
    if duration is not None:
        for label, window in (("15m", "15min"), ("1h", "1h"), ("4h", "4h")):
            steps = int(round(pd.Timedelta(window) / duration))
            if steps >= 1:
                changes[f"change_{label}_pct"] = _pct_change(df, steps)
    return changes


def compute_dxy_context(df: pd.DataFrame | None, gauge: dict | None) -> dict | None:
    """Deterministic DXY context block for the canonical snapshot.

    Returns None only when there is no DXY information at all (no candles,
    no gauge). The DXY gate still works on the gauge alone; this context
    only enriches the picture around it.
    """
    if (df is None or df.empty) and gauge is None:
        return None

    ctx: dict = {
        "kind": "dxy_context",
        "source": "intraday candles",
        "ts": str(datetime.now(timezone.utc)),
        "level": None,
        "direction": None,
        "momentum_pct": None,
        "trend": None,
        "trend_adx": None,
        "volatility_pct": None,
        "score": None,
        "classification": None,
        "detail": "",
    }
    ctx.update({f"change_{k}_pct": None for k in ("15m", "1h", "4h")})

    if df is not None and not df.empty and len(df) >= 2:
        close = df["close"]
        ctx["level"] = round(float(close.iloc[-1]), 4)
        roc20 = _pct_change(df, 19)
        if roc20 is not None:
            ctx["direction"] = "up" if roc20 > _DXY_DIRECTION_THRESHOLD_PCT else "down" if roc20 < -_DXY_DIRECTION_THRESHOLD_PCT else "flat"
        ctx["momentum_pct"] = _pct_change(df, 4)
        if len(df) >= 50:
            e20, e50 = ema(close, 20), ema(close, 50)
            if float(e20.iloc[-1]) > float(e50.iloc[-1]):
                ctx["trend"] = "bull"
            elif float(e20.iloc[-1]) < float(e50.iloc[-1]):
                ctx["trend"] = "bear"
            else:
                ctx["trend"] = "flat"
            ctx["trend_adx"] = round(float(adx(df).iloc[-1]), 2)
        ctx["volatility_pct"] = round(float(atr(df).iloc[-1]) / float(close.iloc[-1]) * 100, 4) if float(close.iloc[-1]) else None
        ctx.update(_changes(df))
        ctx["source"] = "intraday candles"
        ctx["detail"] = f"DXY {ctx['direction']} ({ctx['momentum_pct']}% momentum), trend {ctx['trend']}"

    # Score: start from the gauge (weak-dollar = high) and adjust for the
    # intraday direction/momentum/trend. All adjustments are bounded and
    # deterministic — no AI involvement.
    base = float(gauge["value"]) if gauge else 50.0
    score = base
    if ctx["direction"] == "down":
        score += 10
    elif ctx["direction"] == "up":
        score -= 10
    if ctx["momentum_pct"] is not None:
        score += max(-15.0, min(15.0, -ctx["momentum_pct"] * 5.0))
    if ctx["trend"] == "bear":
        score += 10
    elif ctx["trend"] == "bull":
        score -= 10
    ctx["score"] = int(round(max(0.0, min(100.0, score))))
    if ctx["level"] is None:
        ctx["source"] = "gauge only"
        ctx["detail"] = "no intraday DXY candles; gauge-only interpretation"
    ctx["classification"] = (
        "Bullish (USD weak)" if ctx["score"] >= 60 else "Bearish (USD strong)" if ctx["score"] <= 40 else "Neutral"
    )
    return ctx
