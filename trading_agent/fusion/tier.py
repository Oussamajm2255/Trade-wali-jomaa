"""Confidence tiers + A+/A/NO TRADE labels (V-MONSTER §58/§59/§81).

Tier assignment (spec §58) turns the calibrated confidence into three
actionable bands:

- HIGH: the calibrated win rate (historical outcomes in the signal's
  own confidence bucket) is at least `tier_high_min_calibrated`. Only a
  calibrated number can claim this — unvalidated history is never HIGH.
- LOW: the calibrated win rate is below `tier_low_max_calibrated`. A
  sufficient sample that says the bucket loses money refuses the
  proposal (the risk engine's LOW tier gate).
- MEDIUM: everything else — uncalibrated signals included. Honest,
  unvalidated data is not a block and not a promotion (spec §4/§21).

Size caps (spec §59): HIGH trades the full size, MEDIUM is capped at
`tier_medium_size_cap`; the absolute position/exposure limits are
unchanged. LOW never reaches sizing (rejected first).

Signal labels (spec §81): A+ requires the top of EVERY axis — HIGH
statistical tier, structural quality, liquidity room and timing. A is
any approved proposal that is not A+; NO TRADE is a rejection. The
label is deterministic: same inputs, same label.
"""
from __future__ import annotations

from enum import StrEnum

NO_TRADE_LABEL = "NO TRADE"


class ConfidenceTier(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def assign_tier(calibrated: float | None, settings) -> ConfidenceTier:
    """Confidence tier from the calibrated win rate; uncalibrated = MEDIUM."""
    if calibrated is None:
        return ConfidenceTier.MEDIUM
    if calibrated >= float(getattr(settings, "tier_high_min_calibrated", 0.6) or 0.6):
        return ConfidenceTier.HIGH
    if calibrated < float(getattr(settings, "tier_low_max_calibrated", 0.45) or 0.45):
        return ConfidenceTier.LOW
    return ConfidenceTier.MEDIUM


def tier_size_mult(tier: ConfidenceTier, settings) -> float:
    """Size fraction by tier: HIGH full, MEDIUM capped, LOW none."""
    if tier == ConfidenceTier.HIGH:
        return 1.0
    if tier == ConfidenceTier.LOW:
        return 0.0
    return min(
        1.0, float(getattr(settings, "tier_medium_size_cap", 0.75) or 0.75)
    )


def signal_label(
    tier: ConfidenceTier,
    setup_score: float | None,
    timing: dict | None,
    settings,
) -> str:
    """A+ = top structural + liquidity + timing + statistical bucket."""
    if tier != ConfidenceTier.HIGH:
        return "A"
    if setup_score is None or setup_score < float(
        getattr(settings, "a_plus_setup_quality_min", 0.7) or 0.7
    ):
        return "A"
    timing = timing or {}
    quality = timing.get("quality")
    if not isinstance(quality, (int, float)) or quality < float(
        getattr(settings, "a_plus_timing_min", 0.7) or 0.7
    ):
        return "A"
    room_r = timing.get("room_r")
    if not isinstance(room_r, (int, float)) or room_r < float(
        getattr(settings, "a_plus_room_min_r", 2.0) or 2.0
    ):
        return "A"
    return "A+"
