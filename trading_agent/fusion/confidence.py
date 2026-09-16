"""Confidence calibration infrastructure (spec §21).

raw_confidence = abs(direction_score) stays exactly as it was (§16),
and is never presented as a probability. calibrated_confidence is
estimated from historical *resolved signals* only (spec §21: the win
rate inside the signal's own raw-confidence bucket): it returns None
until enough outcomes exist (calibration_min_samples in total and
calibration_min_per_bin in the signal's own confidence bucket), and the
product never claims calibration before that.

The outcome engine (phase 5) fills SignalRecord.outcome; this module
is the read side of that pipeline and is inert until then.
"""
from __future__ import annotations

import logging

from trading_agent.config import Settings
from trading_agent.store import actions

logger = logging.getLogger(__name__)

_BIN_WIDTH = 0.05  # raw-confidence bucket width


def calibrated_confidence(raw: float, settings: Settings) -> float | None:
    """Calibrate a raw confidence from evaluated outcomes, or None.

    Calibration = the historical win rate inside the signal's own
    confidence bucket, guarded by sample-size minimums. Returns None
    (never a fake number) when the data does not yet support a claim.
    """
    try:
        rows = actions.evaluated_signal_outcomes(limit=settings.calibration_window)
    except Exception as exc:  # noqa: BLE001 - calibration must never kill a cycle
        logger.warning("confidence calibration read failed: %s", exc)
        return None
    total = len(rows)
    if total < settings.calibration_min_samples:
        return None

    # Bucket the raw confidence; [0.05, 0.10) etc. Float division is
    # rounded before truncation (0.7 / 0.05 is 13.999... in binary), and
    # the index is capped so 1.0 lands in the top [0.95, 1.0) bucket
    # instead of its own empty one.
    idx = min(19, int(round(raw / _BIN_WIDTH, 6)))
    bucket = round(idx * _BIN_WIDTH, 2)
    upper = round(bucket + _BIN_WIDTH, 2)
    matched = [
        row for row in rows
        if row["confidence"] is not None and bucket <= row["confidence"] < upper
    ]
    if len(matched) < settings.calibration_min_per_bin:
        return None
    wins = sum(1 for row in matched if row["correct"])
    return round(wins / len(matched), 4)
