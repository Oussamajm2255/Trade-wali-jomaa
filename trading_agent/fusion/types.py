"""Fusion-layer schemas (spec §16-§21): direction vs quality separation.

The fusion layer produces several independently inspectable numbers —
never one opaque confidence. The risk engine consumes FusionContext
alongside its hard gates; nothing here overrides the risk engine.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from trading_agent.schema.types import Side


class ConflictState(StrEnum):
    """Overall contradiction state (spec §19)."""

    ALIGNED = "ALIGNED"
    MIXED = "MIXED"
    CONFLICTED = "CONFLICTED"


class NoTradeReason(StrEnum):
    """Why the robot says NO TRADE (spec §20).

    Not every reason maps to an active gate: NEWS_RISK, INSUFFICIENT_DATA
    and RISK_LIMIT are defined for completeness and consumed by the
    existing data-quality / risk gates respectively.
    """

    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_SETUP_QUALITY = "LOW_SETUP_QUALITY"
    MTF_CONFLICT = "MTF_CONFLICT"
    REGIME_CONFLICT = "REGIME_CONFLICT"
    DXY_CONFLICT = "DXY_CONFLICT"
    STRUCTURE_CONFLICT = "STRUCTURE_CONFLICT"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    BAD_SPREAD = "BAD_SPREAD"
    ABNORMAL_SPEED = "ABNORMAL_SPEED"
    INSUFFICIENT_ROOM = "INSUFFICIENT_ROOM"
    NEWS_RISK = "NEWS_RISK"
    SHOCK = "SHOCK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    STATISTICAL_EDGE_UNKNOWN = "STATISTICAL_EDGE_UNKNOWN"
    STATISTICAL_QUALITY = "STATISTICAL_QUALITY"
    RISK_LIMIT = "RISK_LIMIT"


class FusionResult(BaseModel):
    """Weighted fusion of agent verdicts (spec §16).

    direction_score is the signed raw score in -1..1; raw_confidence is
    its magnitude (spec §21: this is the pre-phase-4 `confidence` and
    must never be called a probability). contributions exposes every
    agent's weighted slice so the system can explain why a signal
    scored highly.
    """

    side: Side
    direction_score: float = Field(ge=-1.0, le=1.0)
    raw_confidence: float = Field(ge=0.0, le=1.0)
    contributions: dict[str, float] = Field(default_factory=dict)


class SetupQuality(BaseModel):
    """Deterministic setup-quality verdict (spec §18), 0..1.

    Every component is transparent and computed from the canonical
    snapshot — the LLM has no input here. Direction is deliberately NOT
    a component (§17): strong directional agreement can still be a poor
    trade.
    """

    score: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)
    detail: str = ""


class Conflict(BaseModel):
    """One deterministic contradiction between the fused side and a
    deterministic context axis (spec §19)."""

    axis: str  # "mtf" | "regime" | "dxy" | "structure"
    no_trade_reason: NoTradeReason
    detail: str


class ConflictReport(BaseModel):
    """Contradiction verdict: ALIGNED / MIXED / CONFLICTED (spec §19)."""

    state: ConflictState
    conflicts: list[Conflict] = Field(default_factory=list)
    conflict_score: float = Field(ge=0.0, le=1.0)

    @property
    def dominant_reason(self) -> NoTradeReason | None:
        """The first (most decisive) conflict's no-trade reason."""
        return self.conflicts[0].no_trade_reason if self.conflicts else None


class FusionContext(BaseModel):
    """Everything the risk engine needs from the fusion layer (§16-§21)."""

    fusion: FusionResult
    setup_quality: SetupQuality
    conflict: ConflictReport
    calibrated_confidence: float | None = None
    # Deterministic labels for the no-trade gates (§20).
    regime: str | None = None  # deterministic regime engine label
    spread_pct: float | None = None  # broker/gauge spread when known
    # Phase D (V-MONSTER §30): TRIGGER_QUALITY vs TRIGGER_SPEED,
    # computed per side in build_fusion_context and stored with every
    # signal record.
    trigger: dict = Field(default_factory=dict)
