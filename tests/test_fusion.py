"""Fusion layer tests (spec §16-§21): the fusion result shape and the
deterministic fusion context assembled for the risk engine."""
from __future__ import annotations

import pytest

from trading_agent.config import Settings
from trading_agent.fusion.engine import build_fusion_context, fuse_verdicts
from trading_agent.fusion.setup_quality import WEIGHTS
from trading_agent.fusion.types import ConflictState
from trading_agent.schema.types import AgentVerdict, Side


class Snap:
    """Duck-typed MarketSnapshot stand-in for the fusion context."""

    def __init__(self, **kw) -> None:
        self.price = kw.get("price", 2000.0)
        self.entry_timeframe = "15m"
        self.indicators = {"15m": kw.get("indicators", {})}
        self.alignment = kw.get("alignment", {})
        self.structure = kw.get("structure", {})
        self.regimes = {"15m": kw.get("regime", {})}
        self.biases = kw.get("biases", {})
        self.dxy = kw.get("dxy")
        self.dxy_context = kw.get("dxy_context")
        self.session_context = kw.get("session_context", {})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        weight_technical=0.45,
        weight_regime=0.35,
        weight_sentiment=0.20,
        side_threshold=0.25,
        conflict_conflicted_min=2,
        conflict_penalty_per_axis=0.15,
        conflict_max_penalty=0.4,
        conflict_block_conflicted=True,
        setup_quality_min=0.45,
    )


def v(name: str, payload: dict) -> AgentVerdict:
    return AgentVerdict(agent=name, model="test", payload=payload)


def _bullish() -> dict[str, AgentVerdict]:
    return {
        "technical": v("technical", {"bias": "long", "conviction": 0.8}),
        "regime": v("regime", {"regime": "trending_up", "trend_strength": 0.8}),
        "sentiment": v("sentiment", {"score": 0.6}),
    }


def test_fusion_result_is_not_one_opaque_number(settings: Settings) -> None:
    fusion = fuse_verdicts(_bullish(), settings)
    assert fusion.side == Side.LONG
    assert fusion.direction_score == pytest.approx(0.45 * 0.8 + 0.35 * 0.8 + 0.20 * 0.6)
    assert fusion.raw_confidence == pytest.approx(abs(fusion.direction_score))
    assert fusion.contributions == {
        "technical": pytest.approx(0.36),
        "regime": pytest.approx(0.28),
        "sentiment": pytest.approx(0.12),
    }


def test_direction_score_is_capped_at_one(settings: Settings) -> None:
    verdicts = {
        "technical": v("technical", {"bias": "long", "conviction": 1.0}),
        "regime": v("regime", {"regime": "trending_up", "trend_strength": 1.0}),
        "sentiment": v("sentiment", {"score": 1.0}),
    }
    fusion = fuse_verdicts(verdicts, settings)
    assert fusion.direction_score == 1.0
    assert fusion.raw_confidence == 1.0


def test_build_fusion_context_on_clean_setup(settings: Settings) -> None:
    snap = Snap(
        alignment={"alignment": "BULLISH_ALIGNMENT"},
        structure={"bos": [{"type": "BOS_BULLISH"}], "support": [1960.0]},
        regime={"regime": "trend_up"},
        biases={"4h": {"bias": "bull"}},
        dxy={"kind": "dxy", "value": 70},
        session_context={"label": "LONDON_NY_OVERLAP"},
        indicators={"atr_14": 10.0, "atr_percentile_100": 0.5},
    )
    ctx = build_fusion_context(snap, _bullish(), settings)
    assert ctx.fusion.side == Side.LONG
    assert ctx.conflict.state == ConflictState.ALIGNED
    assert ctx.setup_quality.score > 0.8
    assert ctx.calibrated_confidence is None  # no outcomes yet (§21)
    assert ctx.regime == "trend_up"


def test_mixed_conflict_reduces_setup_quality(settings: Settings) -> None:
    # Deterministic regime contradicts the fused LONG: 1 axis = MIXED.
    snap = Snap(
        alignment={"alignment": "BULLISH_ALIGNMENT"},
        structure={"bos": [{"type": "BOS_BULLISH"}], "support": [1960.0]},
        regime={"regime": "trend_down"},
        biases={"4h": {"bias": "bull"}},
        dxy={"kind": "dxy", "value": 70},
        session_context={"label": "LONDON_NY_OVERLAP"},
        indicators={"atr_14": 10.0, "atr_percentile_100": 0.5},
    )
    ctx = build_fusion_context(snap, _bullish(), settings)
    assert ctx.conflict.state == ConflictState.MIXED
    assert "conflict penalty -0.15" in ctx.setup_quality.detail
    # The unpenalised score would be higher by exactly one axis penalty.
    clean = build_fusion_context(
        Snap(
            alignment={"alignment": "BULLISH_ALIGNMENT"},
            structure={"bos": [{"type": "BOS_BULLISH"}], "support": [1960.0]},
            regime={"regime": "trend_up"},
            biases={"4h": {"bias": "bull"}},
            dxy={"kind": "dxy", "value": 70},
            session_context={"label": "LONDON_NY_OVERLAP"},
            indicators={"atr_14": 10.0, "atr_percentile_100": 0.5},
        ),
        _bullish(),
        settings,
    )
    assert ctx.setup_quality.score == pytest.approx(
        clean.setup_quality.score
        - WEIGHTS["regime"] * (1.0 - 0.2)  # regime component: trend_up vs trend_down
        - 0.15  # the MIXED conflict penalty
    )


def test_conflicted_without_block_takes_capped_penalty(settings: Settings) -> None:
    settings.conflict_block_conflicted = False
    snap = Snap(
        alignment={"alignment": "BULLISH_ALIGNMENT"},
        structure={"bos": [{"type": "BOS_BULLISH"}], "support": [1960.0]},
        regime={"regime": "trend_down"},  # conflict 1
        biases={"4h": {"bias": "bear"}},  # conflict 2 -> CONFLICTED
        dxy={"kind": "dxy", "value": 70},
        session_context={"label": "LONDON_NY_OVERLAP"},
        indicators={"atr_14": 10.0, "atr_percentile_100": 0.5},
    )
    ctx = build_fusion_context(snap, _bullish(), settings)
    assert ctx.conflict.state == ConflictState.CONFLICTED
    # 2 axes * 0.15 = 0.30, under the 0.4 cap.
    assert "conflict penalty -0.3" in ctx.setup_quality.detail


def test_conflicted_with_block_keeps_quality_untouched(settings: Settings) -> None:
    snap = Snap(
        regime={"regime": "trend_down"},
        biases={"4h": {"bias": "bear"}},
        dxy={"kind": "dxy", "value": 70},
    )
    ctx = build_fusion_context(snap, _bullish(), settings)
    assert ctx.conflict.state == ConflictState.CONFLICTED
    # Blocking is the risk engine's job; quality is reported unpenalised.
    assert "conflict penalty" not in ctx.setup_quality.detail
