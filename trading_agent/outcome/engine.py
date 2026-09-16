"""Trade outcome engine (spec §23).

For every completed opportunity the engine classifies the result
(WIN / LOSS / BREAKEVEN / EXPIRED / INVALIDATED) and computes the
trade's own statistics: R multiple, MFE/MAE (price and R), bars held,
time-to-stop and time-to-target. Win/loss alone is never the story.

On close it also closes the loop opened by phase 4: the signal record
gets its outcome, and every tracked agent verdict of that cycle gets
`actual_outcome` / `correct` — the read side of the confidence
calibration infrastructure (spec §21).
"""
from __future__ import annotations

from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_agent.schema.types import Side
from trading_agent.store.models import AgentTrack, Position, SignalRecord


class Outcome(StrEnum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


# Exit reasons that fix the outcome regardless of realised PnL.
_FIXED_OUTCOME = {"expiry": Outcome.EXPIRED, "invalidated": Outcome.INVALIDATED}


def classify_outcome(exit_reason: str | None, pnl: float | None) -> Outcome | None:
    """Classify a closed trade (spec §23).

    EXPIRED/INVALIDATED follow the exit reason; every other exit —
    stop-loss, take-profit, manual or broker close — is judged by its
    realised PnL. A trailing stop that closes above entry is a WIN, a
    slipped target that loses money is a LOSS. `None` = unresolved
    (no PnL yet), never a fake classification.
    """
    if exit_reason in _FIXED_OUTCOME:
        return _FIXED_OUTCOME[exit_reason]
    if pnl is None:
        return None
    if pnl > 0:
        return Outcome.WIN
    if pnl < 0:
        return Outcome.LOSS
    return Outcome.BREAKEVEN


def r_multiple(pnl: float | None, entry: float | None, stop: float | None, size: float | None) -> float | None:
    """PnL expressed in units of the initial risk (|entry - stop| * size)."""
    if pnl is None or not entry or not stop or not size:
        return None
    risk = abs(entry - stop) * size
    if risk <= 0:
        return None
    return round(pnl / risk, 4)


def excursion_r(pos: Position, price: float | None) -> float | None:
    """A price excursion (MFE/MAE level) in R multiples of the initial risk."""
    if price is None or not pos.entry or not pos.stop:
        return None
    risk_distance = abs(pos.entry - pos.stop)
    if risk_distance <= 0:
        return None
    direction = 1.0 if pos.side == Side.LONG.value else -1.0
    return round((price - pos.entry) * direction / risk_distance, 4)


def time_to_exit(pos: Position) -> tuple[int | None, int | None]:
    """(time_to_stop, time_to_target) in bars: bars held when the exit was
    a stop-out / target hit, else None for the axis that never resolved."""
    if pos.exit_reason == "stop_loss":
        return pos.bars_open, None
    if pos.exit_reason == "take_profit":
        return None, pos.bars_open
    return None, None


def position_metrics(pos: Position) -> dict:
    """All outcome statistics of one closed position (spec §23)."""
    time_to_stop, time_to_target = time_to_exit(pos)
    return {
        "position_id": pos.id,
        "outcome": pos.outcome,
        "r_multiple": pos.r_multiple,
        "mfe_price": pos.mfe_price,
        "mae_price": pos.mae_price,
        "mfe_r": excursion_r(pos, pos.mfe_price),
        "mae_r": excursion_r(pos, pos.mae_price),
        "bars_open": pos.bars_open,
        "time_to_stop": time_to_stop,
        "time_to_target": time_to_target,
    }


def finalize_position(session: Session, pos: Position) -> None:
    """Classify a closed position and propagate the outcome (spec §23).

    Runs inside the caller's session/transaction:
    1. Position gets outcome + R multiple.
    2. Its signal record (via proposal_id) gets the same outcome — the
       rejected-vs-taken memory now knows how every decision resolved.
    3. Every tracked agent verdict of that cycle gets actual_outcome and
       correct (WIN -> True, LOSS -> False, ambiguous -> None), feeding
       the confidence calibration read side (spec §21).
    """
    outcome = classify_outcome(pos.exit_reason, pos.pnl)
    pos.outcome = outcome.value if outcome else None
    pos.r_multiple = r_multiple(pos.pnl, pos.entry, pos.stop, pos.size)

    signal = session.scalar(select(SignalRecord).where(SignalRecord.proposal_id == pos.proposal_id))
    if signal is None:
        return
    signal.outcome = pos.outcome
    signal.r_multiple = pos.r_multiple

    correct: bool | None = None
    if outcome == Outcome.WIN:
        correct = True
    elif outcome == Outcome.LOSS:
        correct = False
    for track in session.scalars(
        select(AgentTrack).where(AgentTrack.signal_id == signal.signal_id)
    ):
        track.actual_outcome = pos.outcome
        track.correct = correct
