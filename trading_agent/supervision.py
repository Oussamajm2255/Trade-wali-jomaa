"""Live signal supervision (V-MONSTER §62).

Once a proposal is sent to the human, the market keeps moving. This
module classifies what a pending proposal means at the current price:

- VALID: the entry is still available — approve as planned.
- DO_NOT_CHASE: the price left the execution zone (beyond the
  max-chase half-width). The setup may still be good, but entering at
  market would chase — wait for a pullback or reject.
- INVALIDATED: the stop or the target was reached — the setup no
  longer exists (entering now would be a different trade).
- EXPIRED: the actionability deadline passed.

The classification is a pure function of (side, entry, stop, target,
price, deadline, max chase), so it is fully unit-testable and usable
in backtests. The loop wires it per tick: `supervise_pending` persists
the state and returns only the proposals whose state CHANGED, which is
exactly the dedup rule for Telegram follow-ups (no spam — spec §62).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import StrEnum

from trading_agent.schema.types import Side
from trading_agent.store import actions

logger = logging.getLogger(__name__)

# Supervision may not import the orchestrator (cycle); the proposal
# rows it reads come straight from the store.


class SupervisionState(StrEnum):
    VALID = "VALID"
    DO_NOT_CHASE = "DO_NOT_CHASE"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


# States that end the proposal's life: approving an invalidated or
# expired signal is refused (the row stops being `pending`).
_TERMINAL = {SupervisionState.INVALIDATED, SupervisionState.EXPIRED}


def supervise_proposal(
    side: Side | str,
    entry: float,
    stop: float,
    target: float,
    price: float,
    *,
    deadline: datetime | None = None,
    max_chase: float = 0.0,
    now: datetime | None = None,
) -> dict:
    """Classify a pending proposal at the current price.

    Priority: EXPIRED > INVALIDATED > DO_NOT_CHASE > VALID. A price
    between stop and entry (pullback) is still VALID — the entry is
    not worse than planned. No deadline never expires (spec §4: absent
    data does not block).
    """
    now = now or datetime.now(timezone.utc)
    if deadline is not None:
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if now >= deadline:
            return {
                "state": SupervisionState.EXPIRED.value,
                "detail": f"deadline exceeded at {deadline.isoformat()}",
                "price": price,
            }

    if side in (Side.LONG, Side.LONG.value):
        if price <= stop:
            return {
                "state": SupervisionState.INVALIDATED.value,
                "detail": f"price {price:.2f} at/below stop {stop:.2f}",
                "price": price,
            }
        if price >= target:
            return {
                "state": SupervisionState.INVALIDATED.value,
                "detail": f"price {price:.2f} already reached target {target:.2f}",
                "price": price,
            }
        if max_chase > 0 and price > entry + max_chase:
            return {
                "state": SupervisionState.DO_NOT_CHASE.value,
                "detail": (
                    f"price {price:.2f} beyond execution zone "
                    f"(entry {entry:.2f} + chase {max_chase:.2f})"
                ),
                "price": price,
            }
    elif side in (Side.SHORT, Side.SHORT.value):
        if price >= stop:
            return {
                "state": SupervisionState.INVALIDATED.value,
                "detail": f"price {price:.2f} at/above stop {stop:.2f}",
                "price": price,
            }
        if price <= target:
            return {
                "state": SupervisionState.INVALIDATED.value,
                "detail": f"price {price:.2f} already reached target {target:.2f}",
                "price": price,
            }
        if max_chase > 0 and price < entry - max_chase:
            return {
                "state": SupervisionState.DO_NOT_CHASE.value,
                "detail": (
                    f"price {price:.2f} beyond execution zone "
                    f"(entry {entry:.2f} - chase {max_chase:.2f})"
                ),
                "price": price,
            }

    return {
        "state": SupervisionState.VALID.value,
        "detail": f"entry still valid at price {price:.2f}",
        "price": price,
    }


def supervise_pending(
    symbol: str, price: float, *, now: datetime | None = None
) -> list[dict]:
    """Supervise every pending proposal of one symbol at the current price.

    Persists the classification (best-effort, like every store write on
    the observation path) and returns only the proposals whose state
    CHANGED since the last tick — each returned dict is one Telegram
    follow-up:

        {"proposal_id", "symbol", "state", "detail", "previous_state"}

    A proposal that reaches INVALIDATED or EXPIRED is moved out of
    `pending` (approving it is refused). DO_NOT_CHASE stays pending:
    the human may still enter on a pullback — the alert says don't
    chase, not don't trade.
    """
    now = now or datetime.now(timezone.utc)
    changes: list[dict] = []
    proposals = actions.pending_proposals(symbol)
    for proposal in proposals:
        try:
            verdict = supervise_proposal(
                proposal.side,
                proposal.entry,
                proposal.stop,
                proposal.target,
                price,
                deadline=proposal.deadline_at,
                max_chase=proposal.max_chase or 0.0,
                now=now,
            )
            previous, changed = actions.update_supervision(
                proposal.id,
                verdict["state"],
                verdict["detail"],
                terminal=verdict["state"] in _TERMINAL,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001 - supervision must never kill the loop
            logger.warning("supervision failed for %s: %s", proposal.id, exc)
            continue
        # The first classification (None -> VALID) is not a follow-up:
        # the proposal message already said the entry is valid. Real
        # changes (VALID -> DO_NOT_CHASE / INVALIDATED / EXPIRED) are.
        if changed and not (
            previous is None and verdict["state"] == SupervisionState.VALID.value
        ):
            changes.append(
                {
                    "proposal_id": proposal.id,
                    "symbol": proposal.symbol,
                    "state": verdict["state"],
                    "detail": verdict["detail"],
                    "previous_state": previous,
                }
            )
    return changes
