"""Fusion layer entry points (spec §16-§21).

fuse_verdicts keeps the exact 45/35/20 weighted formula of the previous
phases and returns a FusionResult with the per-agent contributions.
build_fusion_context assembles the deterministic quality/conflict/
calibration context the risk engine consumes.
"""
from __future__ import annotations

from trading_agent.config import Settings
from trading_agent.fusion.confidence import calibrated_confidence
from trading_agent.fusion.conflicts import detect_conflicts
from trading_agent.fusion.setup_quality import compute_setup_quality
from trading_agent.fusion.trigger import compute_trigger_quality
from trading_agent.fusion.types import ConflictState, FusionContext, FusionResult
from trading_agent.schema.types import AgentVerdict, Bias, Regime, Side


def _agent_contribution(name: str, payload: dict, settings: Settings) -> float:
    """One agent's signed weighted slice (spec §16 baseline: 45/35/20)."""
    if name == "technical":
        bias = Bias(payload["bias"])
        sign = 1.0 if bias == Bias.LONG else -1.0 if bias == Bias.SHORT else 0.0
        return settings.weight_technical * sign * payload["conviction"]
    if name == "sentiment":
        return settings.weight_sentiment * payload["score"]
    if name == "dxy":
        # Signed score agrees with gold_bias by schema; the sign is taken
        # from the bias so a wrong sign can never count.
        bias = Bias(payload["gold_bias"])
        sign = 1.0 if bias == Bias.LONG else -1.0 if bias == Bias.SHORT else 0.0
        return settings.weight_sentiment * sign * abs(payload["score"])
    # regime: prefer the explicit trend_direction (§13), fall back to the
    # legacy regime enum mapping for older payloads.
    direction = payload.get("trend_direction")
    if direction == "up":
        sign = 1.0
    elif direction == "down":
        sign = -1.0
    elif direction == "flat":
        sign = 0.0
    else:
        regime = Regime(payload["regime"])
        sign = (
            1.0
            if regime == Regime.TRENDING_UP
            else -1.0
            if regime == Regime.TRENDING_DOWN
            else 0.0
        )
    return settings.weight_regime * sign * payload["trend_strength"]


def fuse_verdicts(
    verdicts: dict[str, AgentVerdict], settings: Settings
) -> FusionResult:
    """Weighted fusion (spec §16): signed direction_score + raw_confidence."""
    contributions = {
        name: round(_agent_contribution(name, v.payload, settings), 4)
        for name, v in verdicts.items()
    }
    direction_score = round(
        min(1.0, max(-1.0, sum(contributions.values()))), 4
    )
    if direction_score >= settings.side_threshold:
        side = Side.LONG
    elif direction_score <= -settings.side_threshold:
        side = Side.SHORT
    else:
        side = Side.NEUTRAL
    return FusionResult(
        side=side,
        direction_score=direction_score,
        raw_confidence=round(abs(direction_score), 4),
        contributions=contributions,
    )


def build_fusion_context(
    snapshot, verdicts: dict[str, AgentVerdict], settings: Settings,
    htf_timeframe: str | None = None,
) -> FusionContext:
    """Assemble the deterministic fusion context for the risk engine.

    Conflict policy (§19): MIXED reduces the setup-quality score by a
    configurable per-axis penalty; CONFLICTED is left for the risk
    engine to block (conflict_block_conflicted) — and when blocking is
    disabled the quality score takes the capped penalty instead.
    """
    fusion = fuse_verdicts(verdicts, settings)
    quality = compute_setup_quality(fusion.side, snapshot, settings)
    conflict = detect_conflicts(fusion.side, snapshot, settings, htf_timeframe)

    penalty_axes = len(conflict.conflicts)
    if penalty_axes and (
        conflict.state == ConflictState.MIXED or not settings.conflict_block_conflicted
    ):
        penalty = min(
            settings.conflict_max_penalty,
            round(settings.conflict_penalty_per_axis * penalty_axes, 4),
        )
        if penalty:
            quality.score = round(max(0.0, quality.score - penalty), 4)
            quality.detail = f"{quality.detail}; conflict penalty -{penalty}"

    regime_label = (
        ((getattr(snapshot, "regimes", {}) or {}).get(
            getattr(snapshot, "entry_timeframe", "")
        )
        or {}).get("regime")
    )
    gauge = getattr(snapshot, "dxy", None) or {}
    spread = gauge.get("spread") if isinstance(gauge, dict) else None
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    spread_pct = round(float(spread) / price * 100.0, 4) if spread and price > 0 else None

    return FusionContext(
        fusion=fusion,
        setup_quality=quality,
        conflict=conflict,
        calibrated_confidence=calibrated_confidence(fusion.raw_confidence, settings),
        regime=regime_label,
        spread_pct=spread_pct,
        trigger=compute_trigger_quality(fusion.side, snapshot, settings),
    )
