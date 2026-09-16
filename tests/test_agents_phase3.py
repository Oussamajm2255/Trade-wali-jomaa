"""Phase 3 agent layer (spec §12-§14): upgraded fallback outputs,
deterministic regime engine mapping and DXY context fallback."""
from __future__ import annotations

from trading_agent.agents.fallback import dxy_fallback, regime_fallback, technical_fallback
from trading_agent.schema.types import Bias, Regime


def test_technical_fallback_carries_setup_fields() -> None:
    snap = {
        "ema20_gt_ema50": True, "ema50_gt_ema200": True, "macd_hist": 0.5,
        "rsi_14": 60.0, "adx_14": 30.0, "bb_lower": 100.0, "bb_upper": 120.0,
        "alignment": {"label": "BULLISH_ALIGNMENT"},
    }
    out = technical_fallback(snap)
    assert out.bias == Bias.LONG
    assert out.setup_type == "ema_alignment"
    assert out.structure_alignment > 0  # reflects BULLISH_ALIGNMENT
    assert "EMA20" in out.invalidating_condition
    assert out.reasoning  # §12 reasoning field is populated


def test_technical_fallback_conflicted_alignment_negative() -> None:
    snap = {
        "ema20_gt_ema50": True, "ema50_gt_ema200": True, "macd_hist": 0.5,
        "rsi_14": 60.0, "adx_14": 30.0,
        "alignment": {"label": "CONFLICTED"},
    }
    out = technical_fallback(snap)
    assert out.structure_alignment == -0.4


def test_regime_fallback_uses_deterministic_engine() -> None:
    snap = {
        "regime": {
            "regime": "trend_up",
            "adx": 32.0,
            "trend_strength": 0.64,
            "atr_ratio_vs_median": 1.1,
        }
    }
    out = regime_fallback(snap)
    assert out.regime == Regime.TRENDING_UP
    assert out.trend_direction == "up"
    assert out.trend_strength == 0.64
    assert out.confidence == 0.64
    assert out.volatility_state == "expanded"


def test_regime_fallback_engine_transition_and_low_vol() -> None:
    out = regime_fallback({"regime": {"regime": "transition", "atr_ratio_vs_median": 0.8}})
    assert out.regime == Regime.RANGING
    assert out.trend_direction == "flat"
    assert out.volatility_state == "normal"
    out = regime_fallback({"regime": {"regime": "low_volatility", "atr_ratio_vs_median": 0.4}})
    assert out.regime == Regime.RANGING
    assert out.volatility_state == "contracted"


def test_regime_fallback_legacy_heuristic_without_engine() -> None:
    snap = {
        "atr_percentile_100": 0.5, "adx_14": 30.0,
        "ema20_gt_ema50": False, "ema50_gt_ema200": False,
    }
    out = regime_fallback(snap)
    assert out.regime == Regime.TRENDING_DOWN
    assert out.trend_direction == "down"


def test_dxy_fallback_weak_dollar_bullish_gold() -> None:
    out = dxy_fallback({"score": 70, "classification": "Bullish (USD weak)"}, None)
    assert out.gold_bias == Bias.LONG
    assert out.score == 0.4
    assert out.confidence > 0.5
    assert out.dxy_state == "Bullish (USD weak)"


def test_dxy_fallback_strong_dollar_bearish_gold() -> None:
    out = dxy_fallback(None, {"value": 30, "classification": "Bearish (USD strong)"})
    assert out.gold_bias == Bias.SHORT
    assert out.score == -0.4


def test_dxy_fallback_neutral_zone_zero_score() -> None:
    out = dxy_fallback({"score": 50, "classification": "Neutral"}, None)
    assert out.gold_bias == Bias.NEUTRAL
    assert out.score == 0.0
    assert out.confidence == 0.0


def test_dxy_fallback_without_any_info() -> None:
    out = dxy_fallback(None, None)
    assert out.gold_bias == Bias.NEUTRAL
    assert out.score == 0.0
    assert out.dxy_state == "unknown"
