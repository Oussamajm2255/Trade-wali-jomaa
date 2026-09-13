"""Schema validation: the LLM can propose anything; types say no."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_agent.schema.types import (
    Bias,
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
