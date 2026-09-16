"""Fusion layer (INTELLIGENCE_V2 — phase 4, spec §16-§21).

Direction and setup quality are separate concepts (§17): fusion produces
the signed direction_score and its magnitude raw_confidence (§16/§21),
while the deterministic Setup Quality Engine (§18), conflict detection
(§19) and confidence calibration (§21) stay independently inspectable.
Everything here is deterministic — the LLM only proposes, it never
scores its own work.
"""

from trading_agent.fusion.confidence import calibrated_confidence
from trading_agent.fusion.conflicts import detect_conflicts
from trading_agent.fusion.engine import build_fusion_context, fuse_verdicts
from trading_agent.fusion.setup_quality import compute_setup_quality
from trading_agent.fusion.types import (
    Conflict,
    ConflictReport,
    ConflictState,
    FusionContext,
    FusionResult,
    NoTradeReason,
    SetupQuality,
)

__all__ = [
    "Conflict",
    "ConflictReport",
    "ConflictState",
    "FusionContext",
    "FusionResult",
    "NoTradeReason",
    "SetupQuality",
    "build_fusion_context",
    "calibrated_confidence",
    "compute_setup_quality",
    "detect_conflicts",
    "fuse_verdicts",
]
