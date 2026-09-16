"""SQLAlchemy 2.0 ORM models.

SQLite by default (zero-setup) with a one-line env change to Postgres —
the schema is portable. UTC timestamps everywhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Proposal(Base):
    """A risk-approved trade proposal awaiting human decision."""

    __tablename__ = "proposals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    side: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float]
    entry: Mapped[float]
    stop: Mapped[float]
    target: Mapped[float]
    size: Mapped[float]
    risk_amount: Mapped[float]
    expected_rr: Mapped[float]
    rationale: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JSON)
    model: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    position: Mapped["Position | None"] = relationship(back_populates="proposal", uselist=False)


class Position(Base):
    """A paper position: open, then closed by stop/target/expiry."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proposal_id: Mapped[str] = mapped_column(ForeignKey("proposals.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))
    size: Mapped[float]
    entry: Mapped[float]
    stop: Mapped[float]
    target: Mapped[float]
    entry_fee: Mapped[float] = mapped_column(Float, default=0.0)
    exit_fee: Mapped[float] = mapped_column(Float, default=0.0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(8), default="open", index=True)
    # Broker ticket for live (MT5) positions; NULL = paper position.
    broker_ticket: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)

    proposal: Mapped[Proposal] = relationship(back_populates="position")


class RiskState(Base):
    """Singleton (id=1) risk/equity state — kill-switch survives restarts."""

    __tablename__ = "risk_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    equity: Mapped[float]
    start_of_day_equity: Mapped[float]
    day: Mapped[str] = mapped_column(String(10))  # "YYYY-MM-DD" UTC
    peak_equity: Mapped[float]
    halted: Mapped[bool] = mapped_column(Boolean, default=False)
    halt_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class AuditLog(Base):
    """Immutable-ish trail of every decision the system makes."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    level: Mapped[str] = mapped_column(String(8), default="INFO")
    event: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[dict] = mapped_column(JSON)


class AgentTrack(Base):
    """Per-agent output history for AI reliability tracking (spec §15).

    Every agent verdict of every cycle lands here with its prediction,
    confidence and the market regime it was made in. `actual_outcome` /
    `correct` stay NULL until the outcome engine (phase 5) fills them.
    Statistics are analysis-only: weights are never modified from this.
    """

    __tablename__ = "agent_track"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    agent: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(8))  # "llm" | "fallback"
    model: Mapped[str] = mapped_column(String(64))
    prediction: Mapped[dict] = mapped_column(JSON)
    market_regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fallback_method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actual_outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)
    correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
