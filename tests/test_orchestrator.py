"""Fusion logic + deterministic fallback behaviour (no network, no LLM)."""
from __future__ import annotations

import pytest

from trading_agent.agents.fallback import regime_fallback, sentiment_fallback, technical_fallback
from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, Regime, Side


def make_orchestrator(**overrides) -> Orchestrator:
    settings = Settings(
        weight_technical=0.45, weight_regime=0.35, weight_sentiment=0.20,
        side_threshold=0.25, **overrides,
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
    side, confidence = orch._fuse(verdicts)
    assert side == Side.LONG
    assert confidence == pytest.approx(0.45 * 0.8 + 0.35 * 0.8 + 0.20 * 0.6)


def test_all_bearish_verdicts_fuse_short() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "short", "conviction": 0.9}),
        "regime": v("regime", {"regime": "trending_down", "trend_strength": 0.9}),
        "sentiment": v("sentiment", {"score": -0.8}),
    }
    side, _ = orch._fuse(verdicts)
    assert side == Side.SHORT


def test_mixed_weak_verdicts_fuse_neutral() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "neutral", "conviction": 0.1}),
        "regime": v("regime", {"regime": "ranging", "trend_strength": 0.2}),
        "sentiment": v("sentiment", {"score": 0.1}),
    }
    side, _ = orch._fuse(verdicts)
    assert side == Side.NEUTRAL


def test_missing_agent_reduces_score_not_crash() -> None:
    orch = make_orchestrator()
    verdicts = {
        "technical": v("technical", {"bias": "long", "conviction": 0.5}),
    }
    side, confidence = orch._fuse(verdicts)
    # 0.45*0.5 = 0.225 < 0.25 threshold -> neutral
    assert side == Side.NEUTRAL
    assert confidence == pytest.approx(0.225)


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
