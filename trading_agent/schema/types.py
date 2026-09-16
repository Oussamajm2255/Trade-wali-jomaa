"""Domain schemas shared across agents, risk, execution and storage.

Every agent output is validated against these schemas before it is allowed
to influence a decision — the LLM can only *propose*, never bypass, types.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Bias(StrEnum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Regime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class TechnicalOutput(BaseModel):
    """Structured verdict of the technical-analysis agent (spec §12).

    The LLM must only reason from supplied data; invalidating_condition
    must state what would prove the idea wrong.
    """

    bias: Bias
    conviction: float = Field(ge=0.0, le=1.0)
    setup_type: str = ""
    structure_alignment: float = Field(default=0.0, ge=-1.0, le=1.0)
    support: float | None = None
    resistance: float | None = None
    notes: str = ""
    reasoning: str = ""
    invalidating_condition: str = ""


class SentimentOutput(BaseModel):
    """Structured verdict of the sentiment agent (Fear & Greed + context)."""

    score: float = Field(ge=-1.0, le=1.0)
    tone: str
    notes: str = ""


class RegimeOutput(BaseModel):
    """Structured verdict of the market-regime agent (spec §13).

    The AI interprets the deterministic regime engine's output — it never
    calculates or invents market data.
    """

    regime: Regime
    trend_direction: Literal["up", "down", "flat"] = "flat"
    trend_strength: float = Field(ge=0.0, le=1.0)
    volatility_state: Literal["expanded", "contracted", "normal"] = "normal"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""
    reasoning: str = ""


class DxyOutput(BaseModel):
    """Structured verdict of the DXY context agent (spec §14).

    score is a signed gold-bullish strength in -1..1 (positive = weak
    dollar = bullish gold). Its sign must agree with gold_bias, and a
    neutral bias forces score to zero — the LLM cannot smuggle a
    directional score under a neutral label.
    """

    gold_bias: Bias
    score: float = Field(ge=-1.0, le=1.0)
    dxy_state: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""
    reasoning: str = ""

    @model_validator(mode="after")
    def _sign_agrees_with_bias(self) -> "DxyOutput":
        if self.gold_bias == Bias.NEUTRAL:
            self.score = 0.0
        elif self.gold_bias == Bias.LONG and self.score <= 0:
            raise ValueError("score must be > 0 when gold_bias is long")
        elif self.gold_bias == Bias.SHORT and self.score >= 0:
            raise ValueError("score must be < 0 when gold_bias is short")
        return self


class AgentVerdict(BaseModel):
    """A validated, provenance-tracked answer from one analysis agent."""

    agent: str
    model: str
    payload: dict
    source: str = "llm"  # "llm" | "fallback"
    # Set when the LLM call failed and the heuristic fallback was used
    # (spec §37): TIMEOUT | INVALID_JSON | API_ERROR | RATE_LIMIT |
    # EMPTY_RESPONSE | UNKNOWN. None = LLM answered or LLM disabled by
    # configuration (a config state, not a failure).
    failure_reason: str | None = None
    generated_at: datetime = Field(default_factory=utcnow)


class SignalProposal(BaseModel):
    """A risk-approved trade proposal awaiting human approval."""

    id: str | None = None
    symbol: str
    timeframe: str
    side: Side
    confidence: float = Field(ge=0.0, le=1.0)
    entry: float
    stop: float
    target: float
    size: float
    risk_amount: float
    expected_rr: float
    rationale: str
    evidence: dict
    model: str
    status: str = "pending"
    created_at: datetime = Field(default_factory=utcnow)


class Rejection(BaseModel):
    """Why the pipeline did not produce a proposal (always logged)."""

    symbol: str
    reason: str
    created_at: datetime = Field(default_factory=utcnow)
