"""Deterministic heuristic fallbacks used when the LLM is unavailable.

These are intentionally simple, transparent rules. They exist so the
pipeline keeps running (clearly labelled as degraded) instead of failing
silently — never as a substitute for the LLM analysis when configured.
"""
from __future__ import annotations

from trading_agent.schema.types import (
    Bias,
    Regime,
    RegimeOutput,
    SentimentOutput,
    TechnicalOutput,
)


def technical_fallback(snap: dict) -> TechnicalOutput:
    """EMA alignment + MACD histogram sign + RSI bounds."""
    bias = Bias.NEUTRAL
    conviction = 0.0
    rsi = snap["rsi_14"]
    aligned_up = snap["ema20_gt_ema50"] and snap["ema50_gt_ema200"]
    aligned_down = (not snap["ema20_gt_ema50"]) and (not snap["ema50_gt_ema200"])
    if aligned_up and snap["macd_hist"] > 0 and rsi < 70:
        bias = Bias.LONG
        conviction = min(0.8, 0.3 + snap["adx_14"] / 100)
    elif aligned_down and snap["macd_hist"] < 0 and rsi > 30:
        bias = Bias.SHORT
        conviction = min(0.8, 0.3 + snap["adx_14"] / 100)
    notes = (
        "Deterministic fallback: EMA20/50/200 alignment, MACD histogram sign and "
        f"RSI bounds. RSI={rsi}, ADX={snap['adx_14']}."
    )
    return TechnicalOutput(
        bias=bias,
        conviction=round(conviction, 2),
        support=snap.get("bb_lower"),
        resistance=snap.get("bb_upper"),
        notes=notes,
    )


def sentiment_fallback(gauge: dict | None) -> SentimentOutput:
    """Map a 0-100 sentiment gauge onto a -1..1 score."""
    if not gauge:
        return SentimentOutput(score=0.0, tone="unknown", notes="Sentiment gauge unavailable; neutral score.")
    value = gauge["value"]
    score = round((value - 50) / 50, 2)
    return SentimentOutput(
        score=score,
        tone=gauge["classification"],
        notes=f"{gauge.get('source', 'sentiment gauge')} = {value} ({gauge['classification']}).",
    )


def regime_fallback(snap: dict) -> RegimeOutput:
    """ADX + ATR percentile + EMA alignment classify the market regime."""
    adx_value = snap["adx_14"]
    atr_pct = snap["atr_percentile_100"]
    if atr_pct >= 0.85:
        regime = Regime.HIGH_VOLATILITY
    elif adx_value >= 25 and snap["ema20_gt_ema50"]:
        regime = Regime.TRENDING_UP
    elif adx_value >= 25 and not snap["ema20_gt_ema50"]:
        regime = Regime.TRENDING_DOWN
    else:
        regime = Regime.RANGING
    strength = round(min(1.0, adx_value / 50), 2)
    return RegimeOutput(
        regime=regime,
        trend_strength=strength,
        notes=f"ADX={adx_value}, ATR percentile={atr_pct}.",
    )
