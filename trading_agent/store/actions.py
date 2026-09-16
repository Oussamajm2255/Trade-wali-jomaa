"""Persistence actions: proposals, decisions, positions, audit trail."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy import func, select

from trading_agent.schema.types import Side, SignalProposal, utcnow
from trading_agent.store.db import session_scope
from trading_agent.store.models import (
    AgentTrack,
    AuditLog,
    Position,
    Proposal,
    RiskState,
    SignalRecord,
)


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
    market_regime, confidence, failure_reason, fallback_method and
    (phase 5) the signal_id of the cycle the verdict belonged to.
    Recording is best-effort: a tracking failure must never kill a cycle.
    """
    if not rows:
        return
    with session_scope() as session:
        session.add_all(
            [
                AgentTrack(
                    ts=row.get("ts") or utcnow(),
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
                    signal_id=row.get("signal_id"),
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


def evaluated_outcomes(agent: str | None = None, limit: int = 500) -> list[dict]:
    """(confidence, correct) pairs for verdicts whose outcome is known.

    Feeds the confidence-calibration infrastructure (spec §21). The
    outcome engine (phase 5) fills `actual_outcome`/`correct`; until it
    does this returns [] and calibrated confidence stays None.
    """
    with session_scope() as session:
        query = select(AgentTrack).where(AgentTrack.correct.is_not(None))
        if agent:
            query = query.where(AgentTrack.agent == agent)
        rows = list(session.scalars(query.order_by(AgentTrack.ts.desc()).limit(limit)))
    return [
        {"confidence": row.confidence, "correct": row.correct}
        for row in rows
        if row.confidence is not None
    ]


# ------------------------------------------------------------------ signals

_SIGNAL_SEQ_RE = re.compile(r"-(?P<seq>\d{6})$")


def next_signal_id(symbol: str, ts: datetime | None = None) -> str:
    """Unique per-cycle signal ID, e.g. XAUUSD-20260916-001482 (spec §22).

    Format: <SYMBOL>-<YYYYMMDD>-<6-digit daily sequence>. The sequence
    restarts each UTC day; historical replays (backtesting) use the
    signal's own timestamp so IDs stay unique and dated correctly.
    """
    ts = ts or utcnow()
    prefix = "".join(ch for ch in symbol.upper() if ch.isalnum())
    stem = f"{prefix}-{ts.strftime('%Y%m%d')}-"
    with session_scope() as session:
        existing = session.scalars(
            select(SignalRecord.signal_id).where(SignalRecord.signal_id.like(f"{stem}%"))
        ).all()
    seq = 1
    for sid in existing:
        match = _SIGNAL_SEQ_RE.search(sid)
        if match:
            seq = max(seq, int(match.group("seq")) + 1)
    return f"{stem}{seq:06d}"


def record_signal(record: dict) -> str:
    """Persist one complete signal record (spec §22).

    Both proposals AND rejections are stored — the robot remembers the
    opportunities it refused, not only the trades it took. Returns the
    signal_id. Recording is best-effort for the caller (orchestrator).
    """
    with session_scope() as session:
        row = SignalRecord(**record)
        session.add(row)
        session.flush()
        return row.signal_id


def get_signal(signal_id: str) -> SignalRecord | None:
    with session_scope() as session:
        return session.get(SignalRecord, signal_id)


def list_signals(
    limit: int = 50,
    symbol: str | None = None,
    decision: str | None = None,
) -> list[SignalRecord]:
    """Recent signal records, optionally filtered (spec §22)."""
    with session_scope() as session:
        query = select(SignalRecord)
        if symbol:
            query = query.where(SignalRecord.symbol == symbol)
        if decision:
            query = query.where(SignalRecord.final_decision == decision)
        return list(session.scalars(query.order_by(SignalRecord.ts.desc()).limit(limit)))


def link_signal_proposal(signal_id: str | None, proposal_id: str) -> None:
    """Attach a persisted proposal to its signal record (after save)."""
    if not signal_id:
        return
    with session_scope() as session:
        row = session.get(SignalRecord, signal_id)
        if row is not None:
            row.proposal_id = proposal_id


def evaluated_signal_outcomes(limit: int = 500) -> list[dict]:
    """(confidence, correct) pairs for RESOLVED signals (spec §21/§23).

    The confidence-calibration infrastructure reads this: the historical
    win rate inside the signal's own raw-confidence bucket. Only WIN/
    LOSS are countable outcomes; BREAKEVEN/EXPIRED/INVALIDATED and
    unresolved signals are excluded. raw_confidence comes from the
    stored fusion dict — never a number the LLM could rewrite later.
    """
    with session_scope() as session:
        rows = session.scalars(
            select(SignalRecord)
            .where(SignalRecord.outcome.in_(["WIN", "LOSS"]))
            .order_by(SignalRecord.ts.desc())
            .limit(limit)
        ).all()
    out: list[dict] = []
    for row in rows:
        fusion = row.fusion if isinstance(row.fusion, dict) else {}
        conf = fusion.get("raw_confidence")
        if conf is None:
            continue
        out.append({"confidence": conf, "correct": row.outcome == "WIN"})
    return out


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
