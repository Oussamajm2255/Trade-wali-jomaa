"""Schema validation: the LLM can propose anything; types say no."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_agent.schema.types import (
    Bias,
    DxyOutput,
    Regime,
    RegimeOutput,
    SentimentOutput,
    SignalProposal,
    TechnicalOutput,
)


def test_technical_output_valid() -> None:
    out = TechnicalOutput(bias="long", conviction=0.7, support=100.0, resistance=120.0)
    assert out.bias == Bias.LONG


def test_conviction_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalOutput(bias="long", conviction=1.5)


def test_unknown_bias_rejected() -> None:
    with pytest.raises(ValidationError):
        TechnicalOutput(bias="sideways", conviction=0.5)


def test_sentiment_score_bounds() -> None:
    with pytest.raises(ValidationError):
        SentimentOutput(score=1.2, tone="greed")
    assert SentimentOutput(score=-0.8, tone="fear").score == -0.8


def test_regime_enum_enforced() -> None:
    with pytest.raises(ValidationError):
        RegimeOutput(regime="moon", trend_strength=0.5)
    out = RegimeOutput(regime="trending_up", trend_strength=0.6)
    assert out.regime == Regime.TRENDING_UP


def test_proposal_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        SignalProposal(
            symbol="BTC/USDT", timeframe="1h", side="long", confidence=1.1,
            entry=1.0, stop=0.9, target=1.2, size=1.0, risk_amount=10.0,
            expected_rr=2.0, rationale="x", evidence={}, model="test",
        )


# --- Phase 3 schemas (spec §12-§14) ---


def test_technical_output_phase3_fields() -> None:
    out = TechnicalOutput(
        bias="short",
        conviction=0.6,
        setup_type="choch",
        structure_alignment=-0.5,
        reasoning="structure broke",
        invalidating_condition="reclaim",
    )
    assert out.setup_type == "choch"
    assert out.structure_alignment == -0.5
    with pytest.raises(ValidationError):
        TechnicalOutput(bias="long", conviction=0.5, structure_alignment=1.5)


def test_regime_output_direction_and_volatility_enforced() -> None:
    out = RegimeOutput(regime="trending_up", trend_strength=0.6, trend_direction="up")
    assert out.trend_direction == "up"
    with pytest.raises(ValidationError):
        RegimeOutput(regime="trending_up", trend_strength=0.6, trend_direction="sideways")
    with pytest.raises(ValidationError):
        RegimeOutput(regime="ranging", trend_strength=0.2, volatility_state="wild")


def test_dxy_output_sign_must_agree_with_bias() -> None:
    out = DxyOutput(gold_bias="long", score=0.6, dxy_state="weak dollar", confidence=0.7)
    assert out.gold_bias == Bias.LONG
    with pytest.raises(ValidationError):
        DxyOutput(gold_bias="long", score=-0.3)
    with pytest.raises(ValidationError):
        DxyOutput(gold_bias="short", score=0.2)
    with pytest.raises(ValidationError):
        DxyOutput(gold_bias="long", score=1.5)


def test_dxy_output_neutral_forces_zero_score() -> None:
    out = DxyOutput(gold_bias="neutral", score=0.9, dxy_state="mixed")
    assert out.score == 0.0
