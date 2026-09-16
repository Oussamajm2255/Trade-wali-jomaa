"""Deterministic conflict detection (spec §19).

The fused direction is checked against every deterministic context axis
in the canonical snapshot. 0 conflicts -> ALIGNED, 1 -> MIXED, and
>= conflict_conflicted_min -> CONFLICTED. The conflict_score is
min(1.0, 0.5 * n) so two contradictions already cap it.

A high AI confidence score can never hide these contradictions: they
are computed from the deterministic snapshot, not from any LLM output,
and the risk engine blocks CONFLICTED signals by default.
"""
from __future__ import annotations

from trading_agent.config import Settings
from trading_agent.fusion.types import (
    Conflict,
    ConflictReport,
    ConflictState,
    NoTradeReason,
)
from trading_agent.schema.types import Side


def detect_conflicts(
    side: Side, snapshot, settings: Settings, htf_timeframe: str | None = None
) -> ConflictReport:
    """List every deterministic contradiction against the fused side."""
    conflicts: list[Conflict] = []
    if side == Side.NEUTRAL:
        return ConflictReport(state=ConflictState.ALIGNED, conflict_score=0.0)

    # --- MTF axis: HTF bias and the multi-TF alignment classification.
    htf_tf = htf_timeframe or settings.htf_timeframe
    bias = ((getattr(snapshot, "biases", {}) or {}).get(htf_tf) or {}).get("bias")
    if bias and (
        (side == Side.LONG and bias == "bear") or (side == Side.SHORT and bias == "bull")
    ):
        conflicts.append(
            Conflict(
                axis="mtf",
                no_trade_reason=NoTradeReason.MTF_CONFLICT,
                detail=f"{htf_tf} bias {bias} opposes {side.value}",
            )
        )
    alignment = getattr(snapshot, "alignment", {}) or {}
    if alignment.get("alignment") == "CONFLICTED" and not conflicts:
        conflicts.append(
            Conflict(
                axis="mtf",
                no_trade_reason=NoTradeReason.MTF_CONFLICT,
                detail=alignment.get("detail", "multi-timeframe conflict"),
            )
        )

    # --- Regime axis: the deterministic regime engine label.
    regime = (
        ((getattr(snapshot, "regimes", {}) or {}).get(
            getattr(snapshot, "entry_timeframe", "")
        )
        or {}).get("regime")
    )
    if (side == Side.LONG and regime == "trend_down") or (
        side == Side.SHORT and regime == "trend_up"
    ):
        conflicts.append(
            Conflict(
                axis="regime",
                no_trade_reason=NoTradeReason.REGIME_CONFLICT,
                detail=f"deterministic regime {regime} opposes {side.value}",
            )
        )

    # --- DXY axis: the dollar must agree with the direction (gold).
    gauge = getattr(snapshot, "dxy", None)
    if gauge and gauge.get("kind") == "dxy":
        value = float(gauge["value"])
        if side == Side.LONG and value < settings.dxy_long_min:
            conflicts.append(
                Conflict(
                    axis="dxy",
                    no_trade_reason=NoTradeReason.DXY_CONFLICT,
                    detail=f"gauge {value:.0f} not weak-dollar for a LONG",
                )
            )
        if side == Side.SHORT and value > settings.dxy_short_max:
            conflicts.append(
                Conflict(
                    axis="dxy",
                    no_trade_reason=NoTradeReason.DXY_CONFLICT,
                    detail=f"gauge {value:.0f} not strong-dollar for a SHORT",
                )
            )

    # --- Structure axis: an active CHoCH or price beyond the last
    # structural level contradicts the side's market-structure story.
    structure = getattr(snapshot, "structure", {}) or {}
    choch = structure.get("choch") or []
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    if side == Side.LONG:
        if any(e.get("type") == "CHOCH_BEARISH" for e in choch):
            conflicts.append(
                Conflict(
                    axis="structure",
                    no_trade_reason=NoTradeReason.STRUCTURE_CONFLICT,
                    detail="active bearish CHoCH opposes a LONG",
                )
            )
        elif structure.get("support") and price < min(structure["support"]):
            conflicts.append(
                Conflict(
                    axis="structure",
                    no_trade_reason=NoTradeReason.STRUCTURE_CONFLICT,
                    detail="price below all known supports opposes a LONG",
                )
            )
    elif side == Side.SHORT:
        if any(e.get("type") == "CHOCH_BULLISH" for e in choch):
            conflicts.append(
                Conflict(
                    axis="structure",
                    no_trade_reason=NoTradeReason.STRUCTURE_CONFLICT,
                    detail="active bullish CHoCH opposes a SHORT",
                )
            )
        elif structure.get("resistance") and price > max(structure["resistance"]):
            conflicts.append(
                Conflict(
                    axis="structure",
                    no_trade_reason=NoTradeReason.STRUCTURE_CONFLICT,
                    detail="price above all known resistances opposes a SHORT",
                )
            )

    n = len(conflicts)
    if n == 0:
        state = ConflictState.ALIGNED
    elif n >= settings.conflict_conflicted_min:
        state = ConflictState.CONFLICTED
    else:
        state = ConflictState.MIXED
    return ConflictReport(
        state=state,
        conflicts=conflicts,
        conflict_score=round(min(1.0, 0.5 * n), 4),
    )
