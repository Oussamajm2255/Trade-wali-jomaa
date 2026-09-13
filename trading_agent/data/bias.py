"""Deterministic higher-timeframe bias — pro multi-timeframe logic.

The entry timeframe (15m) spots the trigger; the analysis timeframe (4h)
decides WHO is allowed to trade: a LONG is only possible in an HTF
uptrend, a SHORT only in an HTF downtrend, and a choppy HTF means no
trade at all ("the trend is your friend"). Computed here, not by the
LLM, so the gate is reproducible and audit-logged like every other
risk gate.
"""
from __future__ import annotations

import pandas as pd

from trading_agent.data.indicators import adx, ema

MIN_CANDLES = 200  # EMA200 needs a decent warm-up on 4h


def compute_htf_bias(df: pd.DataFrame, adx_min: float = 20.0) -> dict:
    """Classify the higher-timeframe trend as bull / bear / neutral.

    - ema50 vs ema200 gives the trend direction
    - close vs ema200 confirms price is on the right side of it
    - ADX >= adx_min proves the trend has strength (else: neutral chop)
    """
    if len(df) < MIN_CANDLES:
        return {
            "bias": "neutral",
            "score": 0,
            "adx": 0.0,
            "detail": f"not enough candles ({len(df)} < {MIN_CANDLES})",
        }
    close = df["close"]
    ema50_now = float(ema(close, 50).iloc[-1])
    ema200_now = float(ema(close, 200).iloc[-1])
    last_close = float(close.iloc[-1])
    adx_now = float(adx(df).iloc[-1])

    score = 0
    score += 1 if ema50_now > ema200_now else -1
    score += 1 if last_close > ema200_now else -1

    if adx_now < adx_min:
        bias = "neutral"
        detail = f"ADX {adx_now:.1f} < {adx_min:.0f}: no trend (chop)"
    elif score == 2:
        bias = "bull"
        detail = f"ema50 > ema200, price > ema200, ADX {adx_now:.1f}"
    elif score == -2:
        bias = "bear"
        detail = f"ema50 < ema200, price < ema200, ADX {adx_now:.1f}"
    else:
        bias = "neutral"
        detail = f"mixed alignment (score {score:+d}, ADX {adx_now:.1f})"

    return {
        "bias": bias,
        "score": score,
        "adx": round(adx_now, 2),
        "ema50": round(ema50_now, 8),
        "ema200": round(ema200_now, 8),
        "close": round(last_close, 8),
        "detail": detail,
    }
