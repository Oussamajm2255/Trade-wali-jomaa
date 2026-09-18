"""Rejected-setup outcome analysis (V-MONSTER §67).

Did we reject correctly? Every rejection that carried a directional
fused score and has a 30-minute post-snapshot (§65) is classified
against how the market actually moved in the would-be direction:

- CORRECT_REJECT: the market did NOT move in the rejected direction
  (move <= 0) — the refusal saved a losing trade.
- WRONG_REJECT: the market moved in the rejected direction by at least
  `rejection_move_threshold_pct` (a percent, e.g. 0.05 = 0.05%) — the
  refused trade would have paid.
- INCONCLUSIVE: it moved in the rejected direction, but by less than
  the threshold — noise, excluded from the quality ratio.

The quality ratio is correct / (correct + wrong); INCONCLUSIVE and
unresolved rejections never inflate it (spec §36 honesty).
"""
from __future__ import annotations

from enum import StrEnum


class RejectionOutcome(StrEnum):
    CORRECT_REJECT = "CORRECT_REJECT"
    WRONG_REJECT = "WRONG_REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


def rejection_outcome(
    direction_score: float,
    entry_price: float,
    future_price: float,
    threshold_pct: float = 0.05,
) -> RejectionOutcome:
    """Classify one rejected signal against its realized 30m move.

    `direction_score` is the signed fusion score (its sign is the
    direction the robot deliberately did not trade). Pure function.
    """
    if entry_price <= 0 or direction_score == 0:
        return RejectionOutcome.INCONCLUSIVE
    move_pct = (future_price - entry_price) / entry_price
    # Sign-flip so `directional_move` is positive when the market went
    # the way the rejected trade would have needed.
    directional_move = move_pct * (1.0 if direction_score > 0 else -1.0)
    if directional_move <= 0:
        return RejectionOutcome.CORRECT_REJECT
    # threshold_pct is a percent; directional_move is a fraction.
    if directional_move < float(threshold_pct) / 100.0:
        return RejectionOutcome.INCONCLUSIVE
    return RejectionOutcome.WRONG_REJECT


def rejection_quality(
    outcomes: list[dict],
    threshold_pct: float = 0.05,
) -> dict:
    """Aggregate classified rejections into the §67 quality stats."""
    counts: dict[str, int] = {}
    for row in outcomes:
        outcome = rejection_outcome(
            row["direction_score"],
            row["entry_price"],
            row["future_price"],
            threshold_pct=threshold_pct,
        )
        counts[outcome.value] = counts.get(outcome.value, 0) + 1
    correct, wrong = counts.get(RejectionOutcome.CORRECT_REJECT.value, 0), counts.get(
        RejectionOutcome.WRONG_REJECT.value, 0
    )
    denom = correct + wrong
    return {
        "correct": correct,
        "wrong": wrong,
        "inconclusive": counts.get(RejectionOutcome.INCONCLUSIVE.value, 0),
        "resolved": len(outcomes),
        # Honest ratio: only decided cases count (spec §36).
        "correct_rate": round(correct / denom, 4) if denom else None,
    }
