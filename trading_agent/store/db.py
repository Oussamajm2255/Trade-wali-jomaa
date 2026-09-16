"""Engine/session management.

The engine is lazily initialised from settings (or an explicit URL in
tests) so importing the package never touches the filesystem or network.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from trading_agent.config import get_settings
from trading_agent.store.models import Base, RiskState

engine: Engine | None = None
SessionLocal: sessionmaker[Session] | None = None


def init_engine(url: str | None = None) -> Engine:
    """Create (or recreate) the engine, tables, and seed the risk state."""
    global engine, SessionLocal
    url = url or get_settings().database_url
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            # A single shared connection so in-memory DBs behave like a real one.
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    engine = create_engine(url, **kwargs)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    Base.metadata.create_all(engine)
    _migrate_sqlite(engine)
    _seed_risk_state()
    return engine


def _migrate_sqlite(engine: Engine) -> None:
    """Lightweight additive migrations for existing SQLite databases."""
    if engine.url.get_backend_name() != "sqlite":
        return
    with engine.begin() as conn:
        # Phase 1: broker ticket for live (MT5) positions.
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(positions)")}
        if columns and "broker_ticket" not in columns:
            conn.exec_driver_sql("ALTER TABLE positions ADD COLUMN broker_ticket INTEGER")
        # Phase 5 (§23): outcome-engine tracking on positions.
        additive = [
            ("mfe_price", "FLOAT"),
            ("mae_price", "FLOAT"),
            ("bars_open", "INTEGER DEFAULT 0"),
            ("outcome", "VARCHAR(16)"),
            ("r_multiple", "FLOAT"),
        ]
        for name, sqltype in additive:
            if columns and name not in columns:
                conn.exec_driver_sql(f"ALTER TABLE positions ADD COLUMN {name} {sqltype}")
        # Phase 5 (§22): cycle link on agent tracks.
        track_columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(agent_track)")}
        if track_columns and "signal_id" not in track_columns:
            conn.exec_driver_sql("ALTER TABLE agent_track ADD COLUMN signal_id VARCHAR(24)")


def _seed_risk_state() -> None:
    settings = get_settings()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with SessionLocal() as session:  # type: ignore[union-attr]
        if session.get(RiskState, 1) is None:
            session.add(
                RiskState(
                    id=1,
                    equity=settings.paper_starting_equity,
                    start_of_day_equity=settings.paper_starting_equity,
                    day=today,
                    peak_equity=settings.paper_starting_equity,
                    halted=False,
                )
            )
            session.commit()


@contextmanager
def session_scope() -> Iterator[Session]:
    if SessionLocal is None:
        init_engine()
    session = SessionLocal()  # type: ignore[misc]
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
