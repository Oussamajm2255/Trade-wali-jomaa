"""Opportunity clustering + signal dedup (V-MONSTER §31/§40/§41/§52/§53).

An OPPORTUNITY is one directional setup anchored at a deterministic
structure event (the most recent BOS in the side's direction, else the
most recent same-direction displacement candle, else the trigger-side
liquidity pool) plus time proximity. The identity is fully derived from
the canonical snapshot, so two cycles looking at the same physical
event produce the same OPPORTUNITY_ID — and dedup is reproducible, not
a guess.

Lifecycle (§31): FORMING (seen, not signalled) -> TRIGGERED (a proposal
was sent) -> EXPIRED (the opportunity aged past its TTL without ever
triggering). The lifecycle is observability/forensics, never a gate by
itself.

Dedup (§52/§53), opt-out via `opportunity_dedup_enabled`:
- a pending proposal for the same opportunity blocks a WEAKER re-signal
  (OPPORTUNITY_ACTIVE) but lets a stronger one through (supersede);
- any proposal for the same opportunity inside the dedup window at a
  close price (within `opportunity_price_tolerance_pct`) is suppressed
  as SIGNAL_DUPLICATE;
- no anchor, no data, or a DB failure never blocks (data honesty, §4).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import select

from trading_agent.fusion.types import NoTradeReason
from trading_agent.schema.types import Side, utcnow
from trading_agent.store.db import session_scope
from trading_agent.store.models import Opportunity, Proposal, SignalRecord

logger = logging.getLogger(__name__)


def _as_utc(dt: datetime) -> datetime:
    """Normalise a possibly-naive SQLite datetime to aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class OpportunityLifecycle(StrEnum):
    FORMING = "FORMING"
    TRIGGERED = "TRIGGERED"
    EXPIRED = "EXPIRED"


def anchor_from_snapshot(snap: Any, side: Side) -> str | None:
    """Deterministic opportunity anchor from the canonical snapshot.

    Order: BOS in the side's direction, then same-direction displacement,
    then the trigger-side liquidity pool (lows for LONG, highs for SHORT).
    None when the side is neutral or nothing maps — dedup then never
    fires (honest, never fabricated).
    """
    if side == Side.NEUTRAL:
        return None
    structure = getattr(snap, "structure", None) or {}

    bos_type = "BOS_BULLISH" if side == Side.LONG else "BOS_BEARISH"
    bos = [e for e in structure.get("bos", []) or [] if e.get("type") == bos_type]
    if bos:
        event = min(bos, key=lambda e: e.get("age", 0))
        return f"bos:{event.get('timestamp')}"

    direction = "bullish" if side == Side.LONG else "bearish"
    disps = [
        e for e in structure.get("displacements", []) or [] if e.get("direction") == direction
    ]
    if disps:
        event = min(disps, key=lambda e: e.get("age", 0))
        return f"disp:{event.get('timestamp')}"

    price = float(getattr(snap, "price", 0.0) or 0.0)
    if price > 0:
        liquidity = getattr(snap, "liquidity", None) or {}
        levels = liquidity.get("levels") or []
        candidates = [
            float(lv["price"])
            for lv in levels
            if isinstance(lv, dict)
            and lv.get("price") is not None
            and (float(lv["price"]) < price if side == Side.LONG else float(lv["price"]) > price)
        ]
        if candidates:
            level = max(candidates) if side == Side.LONG else min(candidates)
            return f"liq:{level}"
    return None


def opportunity_id(symbol: str, timeframe: str, side: Side, anchor: str | None) -> str | None:
    """Stable OPPORTUNITY_ID: symbol + timeframe + direction + anchor."""
    if not anchor:
        return None
    return f"{symbol.upper()}:{timeframe}:{side.value}:{anchor}"


def opportunity_age_minutes(oid: str, now: datetime | None = None) -> float | None:
    """Age of a tracked opportunity in minutes; None when unknown.

    Feeds the Phase G timing lifecycle component (how much of the
    opportunity TTL has been consumed). Best-effort by contract: a
    storage failure returns None and the component scores neutral.
    """
    now = now or utcnow()
    try:
        with session_scope() as session:
            row = session.get(Opportunity, oid)
            if row is None:
                return None
            return max(0.0, (now - _as_utc(row.first_seen_ts)).total_seconds() / 60.0)
    except Exception as exc:  # noqa: BLE001 - timing must never kill the cycle
        logger.warning("opportunity age lookup failed for %s: %s", oid, exc)
        return None


def dedup_verdict(
    symbol: str,
    timeframe: str,
    side: Side,
    anchor: str | None,
    price: float,
    setup_score: float,
    settings: Any,
    now: datetime | None = None,
) -> dict | None:
    """Decide whether this signal duplicates an already-seen opportunity.

    Returns None (allow) or {"no_trade_reason", "detail"} (suppress).
    Never raises: storage problems fail open — a DB hiccup must not
    block the pipeline.
    """
    if not settings.opportunity_dedup_enabled or not anchor:
        return None
    oid = opportunity_id(symbol, timeframe, side, anchor)
    now = now or utcnow()
    window = timedelta(minutes=settings.opportunity_dedup_window_minutes)
    try:
        with session_scope() as session:
            # 1. A pending proposal for the same opportunity: a weaker
            # re-signal is suppressed, a stronger one supersedes (§53).
            pending_ids = list(
                session.scalars(
                    select(Proposal.id).where(
                        Proposal.symbol == symbol, Proposal.status == "pending"
                    )
                )
            )
            if pending_ids:
                pending_rows = session.scalars(
                    select(SignalRecord).where(
                        SignalRecord.proposal_id.in_(pending_ids),
                        SignalRecord.opportunity_id == oid,
                    )
                ).all()
                if pending_rows:
                    best = max(
                        (r.setup_quality or {}).get("score", 0.0) for r in pending_rows
                    )
                    if setup_score < best:
                        return {
                            "no_trade_reason": NoTradeReason.OPPORTUNITY_ACTIVE.value,
                            "detail": (
                                f"opportunity {oid} still active with a stronger "
                                f"signal (setup {setup_score:.2f} < {best:.2f})"
                            ),
                        }
                    return None  # stronger signal: supersede at save time

            # 2. Any proposal for the same opportunity inside the window
            # at a close price is a duplicate (§52). Timestamps compare
            # in Python (SQLite stores naive UTC; other backends aware).
            recent = session.scalars(
                select(SignalRecord)
                .where(
                    SignalRecord.opportunity_id == oid,
                    SignalRecord.final_decision == "proposal",
                )
                .order_by(SignalRecord.ts.desc())
                .limit(20)
            ).all()
        for row in recent:
            if _as_utc(row.ts) < now - window:
                break  # ordered desc: everything further back is older
            snapshot = row.market_snapshot or {}
            prev_price = snapshot.get("price")
            if prev_price is None:
                continue
            pct = abs(float(prev_price) - price) / price * 100.0
            if pct <= settings.opportunity_price_tolerance_pct:
                return {
                    "no_trade_reason": NoTradeReason.SIGNAL_DUPLICATE.value,
                    "detail": (
                        f"opportunity {oid} already signalled "
                        f"{row.signal_id} ({pct:.3f}% away) within the dedup window"
                    ),
                }
        return None
    except Exception as exc:  # noqa: BLE001 - storage must never block the cycle
        logger.warning("opportunity dedup check failed (fails open): %s", exc)
        return None


def track_opportunity(
    oid: str,
    symbol: str,
    timeframe: str,
    side: Side,
    anchor: str,
    triggered: bool,
    trigger_signal_id: str | None,
    settings: Any,
    now: datetime | None = None,
) -> None:
    """Upsert the lifecycle row (§31): FORMING -> TRIGGERED -> EXPIRED.

    Best-effort by contract: a tracking failure is a warning, never a
    cycle failure.
    """
    now = now or utcnow()
    ttl = timedelta(minutes=getattr(settings, "opportunity_ttl_minutes", 720))
    try:
        with session_scope() as session:
            row = session.get(Opportunity, oid)
            if row is None:
                row = Opportunity(
                    opportunity_id=oid,
                    symbol=symbol,
                    timeframe=timeframe,
                    side=side.value,
                    anchor=anchor,
                    state=OpportunityLifecycle.FORMING.value,
                    first_seen_ts=now,
                    last_seen_ts=now,
                )
                session.add(row)
            row.last_seen_ts = now
            if triggered:
                row.state = OpportunityLifecycle.TRIGGERED.value
                row.trigger_signal_id = trigger_signal_id
            elif row.state != OpportunityLifecycle.TRIGGERED.value:
                if now - _as_utc(row.first_seen_ts) > ttl:
                    row.state = OpportunityLifecycle.EXPIRED.value
    except Exception as exc:  # noqa: BLE001 - observability must never kill the cycle
        logger.warning("opportunity tracking failed for %s: %s", oid, exc)
