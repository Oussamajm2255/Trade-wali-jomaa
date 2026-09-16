"""Fusion logic + deterministic fallback behaviour (no network, no LLM)."""
from __future__ import annotations

import pytest

from trading_agent.agents.fallback import regime_fallback, sentiment_fallback, technical_fallback
from trading_agent.agents.orchestrator import Orchestrator, failure_block_reason
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, Regime, Side


def make_orchestrator(**overrides) -> Orchestrator:
    settings = Settings(
        weight_technical=0.45, weight_regime=0.35, weight_sentiment=0.20,
        side_threshold=0.25,
        # Hermetic: never inherit a live DEEPSEEK_API_KEY from .env —
        # the _run_agents tests assert fallback-only verdicts (§37).
        deepseek_api_key=None,
        **overrides,
    )
    # market/risk are unused by _fuse; None is safe here.
    return Orchestrator(settings, market=None, risk=None)  # type: ignore[arg-type]


def v(name: str, payload: dict) -> AgentVerdict:
    return AgentVerdict(agent=name, model="test", payload=payload)


def test_all_bullish_verdicts_fuse_long() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "long", "conviction": 0.8}),
        "regime": v("regime", {"regime": "trending_up", "trend_strength": 0.8}),
        "sentiment": v("sentiment", {"score": 0.6}),
    }
    fusion = orch._fuse(verdicts)
    assert fusion.side == Side.LONG
    assert fusion.direction_score == pytest.approx(0.45 * 0.8 + 0.35 * 0.8 + 0.20 * 0.6)
    assert fusion.raw_confidence == pytest.approx(abs(fusion.direction_score))
    assert set(fusion.contributions) == {"technical", "regime", "sentiment"}


def test_all_bearish_verdicts_fuse_short() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "short", "conviction": 0.9}),
        "regime": v("regime", {"regime": "trending_down", "trend_strength": 0.9}),
        "sentiment": v("sentiment", {"score": -0.8}),
    }
    fusion = orch._fuse(verdicts)
    assert fusion.side == Side.SHORT
    assert fusion.direction_score < 0


def test_mixed_weak_verdicts_fuse_neutral() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "neutral", "conviction": 0.1}),
        "regime": v("regime", {"regime": "ranging", "trend_strength": 0.2}),
        "sentiment": v("sentiment", {"score": 0.1}),
    }
    fusion = orch._fuse(verdicts)
    assert fusion.side == Side.NEUTRAL


def test_missing_agent_reduces_score_not_crash() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "long", "conviction": 0.5}),
    }
    fusion = orch._fuse(verdicts)
    # 0.45*0.5 = 0.225 < 0.25 threshold -> neutral
    assert fusion.side == Side.NEUTRAL
    assert fusion.raw_confidence == pytest.approx(0.225)


def test_technical_fallback_bullish_alignment() -> None:
    snap = {
        "ema20_gt_ema50": True, "ema50_gt_ema200": True, "macd_hist": 0.5,
        "rsi_14": 60.0, "adx_14": 30.0, "bb_lower": 100.0, "bb_upper": 120.0,
    }
    out = technical_fallback(snap)
    assert out.bias.value == "long"
    assert out.conviction > 0


def test_technical_fallback_neutral_when_mixed() -> None:
    snap = {
        "ema20_gt_ema50": True, "ema50_gt_ema200": False, "macd_hist": -0.2,
        "rsi_14": 55.0, "adx_14": 15.0, "bb_lower": 100.0, "bb_upper": 120.0,
    }
    out = technical_fallback(snap)
    assert out.bias.value == "neutral"


def test_sentiment_fallback_mapping() -> None:
    greedy = sentiment_fallback({"value": 80, "classification": "Greed", "ts": "1"})
    assert greedy.score == 0.6
    fearful = sentiment_fallback({"value": 20, "classification": "Fear", "ts": "1"})
    assert fearful.score == -0.6
    missing = sentiment_fallback(None)
    assert missing.score == 0.0


def test_regime_fallback_high_volatility_wins() -> None:
    snap = {
        "atr_percentile_100": 0.95, "adx_14": 10.0,
        "ema20_gt_ema50": True, "ema50_gt_ema200": True,
    }
    out = regime_fallback(snap)
    assert out.regime == Regime.HIGH_VOLATILITY


# --- Phase 3 (spec §12-§14, §37) ---


def test_dxy_verdict_fuses_like_sentiment() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "long", "conviction": 0.8}),
        "regime": v("regime", {"regime": "trending_up", "trend_strength": 0.8}),
        "dxy": v("dxy", {"gold_bias": "long", "score": 0.6}),
    }
    fusion = orch._fuse(verdicts)
    assert fusion.side == Side.LONG
    assert fusion.raw_confidence == pytest.approx(0.45 * 0.8 + 0.35 * 0.8 + 0.20 * 0.6)


def test_regime_trend_direction_flat_overrides_enum() -> None:
    orch = make_orchestrator()
    verdicts = {
        # trend_direction "flat" must neutralise the contribution even
        # though the enum says trending_up (the LLM interprets, but a
        # flat direction is authoritative for the sign).
        "regime": v("regime", {"regime": "trending_up", "trend_strength": 0.9, "trend_direction": "flat"}),
        "technical": v("technical", {"bias": "neutral", "conviction": 0.1}),
        "sentiment": v("sentiment", {"score": 0.0}),
    }
    fusion = orch._fuse(verdicts)
    assert fusion.side == Side.NEUTRAL
    assert fusion.raw_confidence == pytest.approx(0.0)


def test_failure_block_policy() -> None:
    settings = Settings(agent_failure_block_min=2, agent_failure_block_enabled=True)
    assert failure_block_reason([], settings) is None
    assert failure_block_reason(["technical"], settings) is None  # 1 = degrade only
    reason = failure_block_reason(["technical", "regime"], settings)
    assert reason and "blocked" in reason and "technical" in reason
    # Disabled policy never blocks.
    off = Settings(agent_failure_block_min=2, agent_failure_block_enabled=False)
    assert failure_block_reason(["technical", "regime", "dxy"], off) is None


def test_run_agents_routes_dxy_when_context_present() -> None:
    orch = make_orchestrator()  # no API key -> LLM disabled -> fallbacks
    gold_snap = {
        "dxy_context": {"score": 60, "classification": "Bullish (USD weak)"},
        "rsi_14": 55.0,
        "ema20_gt_ema50": True,
        "ema50_gt_ema200": True,
        "macd_hist": 0.1,
        "adx_14": 20.0,
    }
    verdicts = orch._run_agents(gold_snap, None)
    assert "dxy" in verdicts and "sentiment" not in verdicts
    assert all(v.failure_reason is None for v in verdicts.values())


def test_run_agents_keeps_sentiment_without_dxy_context() -> None:
    orch = make_orchestrator()
    crypto_snap = {"rsi_14": 55.0, "adx_14": 20.0}
    verdicts = orch._run_agents(crypto_snap, {"value": 50, "classification": "Neutral"})
    assert "sentiment" in verdicts and "dxy" not in verdicts
