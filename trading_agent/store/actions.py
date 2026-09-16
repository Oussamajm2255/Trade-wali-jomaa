"""Persistence actions: proposals, decisions, positions, audit trail."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from trading_agent.schema.types import Side, SignalProposal, utcnow
from trading_agent.store.db import session_scope
from trading_agent.store.models import AgentTrack, AuditLog, Position, Proposal, RiskState


def audit(level: str, event: str, detail: dict | None = None) -> None:
    with session_scope() as session:
        session.add(AuditLog(level=level, event=event, detail=detail or {}))


def save_proposal(proposal: SignalProposal) -> str:
    """Persist a proposal; supersedes any older pending one for the symbol."""
    with session_scope() as session:
        now = datetime.now(timezone.utc)
        for stale in session.scalars(
            select(Proposal).where(Proposal.symbol == proposal.symbol, Proposal.status == "pending")
        ):
            stale.status = "expired"
            stale.decided_at = now
            stale.decision_note = "superseded by newer proposal"
        row = Proposal(
            symbol=proposal.symbol,
            timeframe=proposal.timeframe,
            side=proposal.side.value,
            confidence=proposal.confidence,
            entry=proposal.entry,
            stop=proposal.stop,
            target=proposal.target,
            size=proposal.size,
            risk_amount=proposal.risk_amount,
            expected_rr=proposal.expected_rr,
            rationale=proposal.rationale,
            evidence=proposal.evidence,
            model=proposal.model,
            status="pending",
        )
        session.add(row)
        session.add(
            AuditLog(
                level="INFO",
                event="proposal_created",
                detail={
                    "symbol": proposal.symbol,
                    "side": proposal.side.value,
                    "confidence": proposal.confidence,
                    "model": proposal.model,
                },
            )
        )
        session.flush()
        return row.id


def get_proposal(proposal_id: str) -> SignalProposal | None:
    with session_scope() as session:
        row = session.get(Proposal, proposal_id)
        return _to_signal(row) if row else None


def list_proposals(status: str | None = None, symbol: str | None = None, limit: int = 20) -> list[SignalProposal]:
    with session_scope() as session:
        query = select(Proposal)
        if status:
            query = query.where(Proposal.status == status)
        if symbol:
            query = query.where(Proposal.symbol == symbol)
        rows = session.scalars(query.order_by(Proposal.created_at.desc()).limit(limit))
        return [_to_signal(row) for row in rows]


def decide_proposal(proposal_id: str, approve: bool, note: str = "") -> SignalProposal | None:
    """Approve or reject a pending proposal. Returns None if not pending."""
    with session_scope() as session:
        row = session.get(Proposal, proposal_id)
        if row is None or row.status != "pending":
            return None
        row.status = "approved" if approve else "rejected"
        row.decided_at = utcnow()
        row.decision_note = note or ("approved via CLI" if approve else "rejected via CLI")
        session.add(
            AuditLog(
                level="INFO",
                event="proposal_approved" if approve else "proposal_rejected",
                detail={"proposal_id": row.id, "symbol": row.symbol, "note": note},
            )
        )
        return _to_signal(row)


def revert_proposal(proposal_id: str, reason: str) -> None:
    """Move an approved proposal back to pending after an execution failure."""
    with session_scope() as session:
        row = session.get(Proposal, proposal_id)
        if row is None or row.status != "approved":
            return
        row.status = "pending"
        row.decided_at = None
        row.decision_note = reason
        session.add(
            AuditLog(
                level="WARNING",
                event="proposal_reverted",
                detail={"proposal_id": proposal_id, "reason": reason},
            )
        )


def open_positions(symbol: str | None = None) -> list[Position]:
    with session_scope() as session:
        query = select(Position).where(Position.status == "open")
        if symbol:
            query = query.where(Position.symbol == symbol)
        return list(session.scalars(query))


def closed_positions(limit: int = 20) -> list[Position]:
    with session_scope() as session:
        query = select(Position).where(Position.status == "closed").order_by(Position.closed_at.desc()).limit(limit)
        return list(session.scalars(query))


def get_risk_state() -> RiskState | None:
    with session_scope() as session:
        row = session.get(RiskState, 1)
        return row


def recent_audit(limit: int = 30) -> list[AuditLog]:
    with session_scope() as session:
        query = select(AuditLog).order_by(AuditLog.ts.desc()).limit(limit)
        return list(session.scalars(query))


def record_agent_track(rows: list[dict]) -> None:
    """Bulk-insert per-agent history rows (spec §15).

    Each row: agent, symbol, timeframe, source, model, prediction,
    market_regime, confidence, failure_reason, fallback_method.
    Recording is best-effort: a tracking failure must never kill a cycle.
    """
    if not rows:
        return
    with session_scope() as session:
        session.add_all(
            [
                AgentTrack(
                    agent=row["agent"],
                    symbol=row.get("symbol", ""),
                    timeframe=row.get("timeframe", ""),
                    source=row.get("source", "llm"),
                    model=row.get("model", ""),
                    prediction=row.get("prediction", {}),
                    market_regime=row.get("market_regime"),
                    confidence=row.get("confidence"),
                    failure_reason=row.get("failure_reason"),
                    fallback_method=row.get("fallback_method"),
                )
                for row in rows
            ]
        )


def agent_reliability(agent: str | None = None, limit: int = 200) -> dict:
    """Rolling per-agent statistics over the last `limit` tracked rows.

    Analysis-only (spec §15): accuracy stays None until the outcome engine
    fills actual outcomes; weights are never modified from this data.
    """
    with session_scope() as session:
        query = select(AgentTrack).order_by(AgentTrack.ts.desc()).limit(limit)
        if agent:
            query = query.where(AgentTrack.agent == agent)
        rows = list(session.scalars(query))
    by_agent: dict[str, dict] = {}
    for row in rows:
        stats = by_agent.setdefault(
            row.agent,
            {
                "total": 0,
                "llm": 0,
                "fallback": 0,
                "failures": 0,
                "evaluated": 0,
                "correct": 0,
                "accuracy": None,
                "avg_confidence": None,
            },
        )
        stats["total"] += 1
        if row.source == "llm":
            stats["llm"] += 1
        else:
            stats["fallback"] += 1
        if row.failure_reason:
            stats["failures"] += 1
        if row.correct is not None:
            stats["evaluated"] += 1
            if row.correct:
                stats["correct"] += 1
    for stats in by_agent.values():
        if stats["evaluated"]:
            stats["accuracy"] = round(stats["correct"] / stats["evaluated"], 4)
        confs = [
            r.confidence
            for r in rows
            if r.confidence is not None and by_agent.get(r.agent) is stats
        ]
        if confs:
            stats["avg_confidence"] = round(sum(confs) / len(confs), 4)
    return {"window": limit, "agents": by_agent}


def _to_signal(row: Proposal) -> SignalProposal:
    return SignalProposal(
        id=row.id,
        symbol=row.symbol,
        timeframe=row.timeframe,
        side=Side(row.side),
        confidence=row.confidence,
        entry=row.entry,
        stop=row.stop,
        target=row.target,
        size=row.size,
        risk_amount=row.risk_amount,
        expected_rr=row.expected_rr,
        rationale=row.rationale,
        evidence=row.evidence or {},
        model=row.model,
        status=row.status,
        created_at=row.created_at,
    )
