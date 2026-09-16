"""Outcome engine (spec §23): trade classification and metrics."""
from trading_agent.outcome.engine import (
    Outcome,
    classify_outcome,
    excursion_r,
    finalize_position,
    position_metrics,
    r_multiple,
    time_to_exit,
)

__all__ = [
    "Outcome",
    "classify_outcome",
    "excursion_r",
    "finalize_position",
    "position_metrics",
    "r_multiple",
    "time_to_exit",
]
