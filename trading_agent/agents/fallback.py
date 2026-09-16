"""Deterministic heuristic fallbacks used when the LLM is unavailable.

These are intentionally simple, transparent rules. They exist so the
pipeline keeps running (clearly labelled as degraded) instead of failing
silently — never as a substitute for the LLM analysis when configured.

Since INTELLIGENCE_V2 phase 2 the fallbacks lean on the deterministic
analysis layer (regime engine, alignment) whenever it is present, so a
degraded cycle still reflects locally computed facts, not guesses.
"""
from __future__ import annotations

from trading_agent.schema.types import (
    Bias,
    DxyOutput,
    Regime,
    RegimeOutput,
    SentimentOutput,
    TechnicalOutput,
)

_ALIGNMENT_MAP = {
    "BULLISH_ALIGNMENT": 0.6,
    "BEARISH_ALIGNMENT": -0.6,
    "CONFLICTED": -0.4,
    "MIXED": 0.0,
}

# Deterministic regime engine labels -> (Regime enum, trend_direction).
_ENGINE_MAP = {
    "trend_up": (Regime.TRENDING_UP, "up"),
    "trend_down": (Regime.TRENDING_DOWN, "down"),
    "range": (Regime.RANGING, "flat"),
    "transition": (Regime.RANGING, "flat"),
    "high_volatility": (Regime.HIGH_VOLATILITY, "flat"),
    "low_volatility": (Regime.RANGING, "flat"),
}


def technical_fallback(snap: dict) -> TechnicalOutput:
    """EMA alignment + MACD histogram sign + RSI bounds (spec §12 shape)."""
    bias = Bias.NEUTRAL
    conviction = 0.0
    rsi = snap.get("rsi_14", 50.0)
    aligned_up = snap.get("ema20_gt_ema50") and snap.get("ema50_gt_ema200")
    aligned_down = (not snap.get("ema20_gt_ema50")) and (not snap.get("ema50_gt_ema200"))
    setup_type = "none"
    invalidating = ""
    if aligned_up and snap.get("macd_hist", 0.0) > 0 and (rsi or 50.0) < 70:
        bias = Bias.LONG
        conviction = min(0.8, 0.3 + (snap.get("adx_14", 0.0) or 0.0) / 100)
        setup_type = "ema_alignment"
        invalidating = "EMA20 crossing back below EMA50 or RSI above 70"
    elif aligned_down and snap.get("macd_hist", 0.0) < 0 and (rsi or 50.0) > 30:
        bias = Bias.SHORT
        conviction = min(0.8, 0.3 + (snap.get("adx_14", 0.0) or 0.0) / 100)
        setup_type = "ema_alignment"
        invalidating = "EMA20 crossing back above EMA50 or RSI below 30"
    # The MTF alignment layer (phase 2) is deterministic — reflect it.
    alignment_label = (snap.get("alignment") or {}).get("label")
    structure_alignment = _ALIGNMENT_MAP.get(alignment_label, 0.0)
    notes = (
        "Deterministic fallback: EMA20/50/200 alignment, MACD histogram sign and "
        f"RSI bounds. RSI={rsi}, ADX={snap.get('adx_14')}."
    )
    return TechnicalOutput(
        bias=bias,
        conviction=round(conviction, 2),
        setup_type=setup_type,
        structure_alignment=round(structure_alignment, 2),
        support=snap.get("bb_lower"),
        resistance=snap.get("bb_upper"),
        notes=notes,
        reasoning=notes,
        invalidating_condition=invalidating,
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
    """Prefer the deterministic regime engine (phase 2); degrade to the
    legacy ADX + ATR percentile heuristic when it is absent."""
    engine = snap.get("regime") or {}
    engine_regime = engine.get("regime")
    if engine_regime in _ENGINE_MAP:
        regime, direction = _ENGINE_MAP[engine_regime]
        strength = engine.get("trend_strength") or min(1.0, (engine.get("adx") or 0.0) / 50)
        ratio = engine.get("atr_ratio_vs_median")
        if ratio is None:
            vol_state = "normal"
        elif ratio >= 1.0:
            vol_state = "expanded"
        elif ratio <= 0.55:
            vol_state = "contracted"
        else:
            vol_state = "normal"
        notes = (
            f"Deterministic regime engine: {engine_regime} (ADX={engine.get('adx')}, "
            f"ATR ratio={ratio})."
        )
        return RegimeOutput(
            regime=regime,
            trend_direction=direction,
            trend_strength=round(min(1.0, strength), 2),
            volatility_state=vol_state,
            confidence=round(min(1.0, strength), 2),
            notes=notes,
            reasoning=notes,
        )

    adx_value = snap.get("adx_14", 0.0)
    atr_pct = snap.get("atr_percentile_100", 0.5)
    if atr_pct >= 0.85:
        regime, direction = Regime.HIGH_VOLATILITY, "flat"
    elif adx_value >= 25 and snap.get("ema20_gt_ema50"):
        regime, direction = Regime.TRENDING_UP, "up"
    elif adx_value >= 25 and not snap.get("ema20_gt_ema50"):
        regime, direction = Regime.TRENDING_DOWN, "down"
    else:
        regime, direction = Regime.RANGING, "flat"
    strength = round(min(1.0, adx_value / 50), 2)
    notes = f"ADX={adx_value}, ATR percentile={atr_pct}."
    return RegimeOutput(
        regime=regime,
        trend_direction=direction,
        trend_strength=strength,
        confidence=strength,
        notes=notes,
        reasoning=notes,
    )


def dxy_fallback(dxy_context: dict | None, gauge: dict | None) -> DxyOutput:
    """Deterministic DXY verdict from the phase-2 context score.

    Keeps the project semantic: 100 = weak dollar = bullish gold.
    """
    value = None
    if dxy_context and dxy_context.get("score") is not None:
        value = float(dxy_context["score"])
    elif gauge:
        value = float(gauge["value"])
    if value is None:
        return DxyOutput(
            gold_bias=Bias.NEUTRAL,
            score=0.0,
            dxy_state="unknown",
            confidence=0.0,
            notes="No DXY information available; neutral score.",
            reasoning="No DXY information available; neutral score.",
        )
    if value >= 55:
        bias = Bias.LONG
    elif value <= 45:
        bias = Bias.SHORT
    else:
        bias = Bias.NEUTRAL
    state = (dxy_context or {}).get("classification") or (gauge or {}).get("classification", "")
    confidence = round(min(1.0, abs(value - 50) / 25), 2)
    notes = (
        f"Deterministic DXY context score={value} ({state}). "
        "100 = weak dollar = bullish gold."
    )
    return DxyOutput(
        gold_bias=bias,
        score=round((value - 50) / 50, 2) if bias != Bias.NEUTRAL else 0.0,
        dxy_state=state,
        confidence=confidence,
        notes=notes,
        reasoning=notes,
    )
