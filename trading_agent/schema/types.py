"""Domain schemas shared across agents, risk, execution and storage.

Every agent output is validated against these schemas before it is allowed
to influence a decision — the LLM can only *propose*, never bypass, types.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


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
    """Structured verdict of the technical-analysis agent."""

    bias: Bias
    conviction: float = Field(ge=0.0, le=1.0)
    support: float | None = None
    resistance: float | None = None
    notes: str = ""


class SentimentOutput(BaseModel):
    """Structured verdict of the sentiment agent (Fear & Greed + context)."""

    score: float = Field(ge=-1.0, le=1.0)
    tone: str
    notes: str = ""


class RegimeOutput(BaseModel):
    """Structured verdict of the market-regime agent."""

    regime: Regime
    trend_strength: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class AgentVerdict(BaseModel):
    """A validated, provenance-tracked answer from one analysis agent."""

    agent: str
    model: str
    payload: dict
    source: str = "llm"  # "llm" | "fallback"
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
