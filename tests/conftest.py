"""Shared fixtures: fresh in-memory DB per test, risk settings."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

import trading_agent.store.db as db
from trading_agent.config import Settings
from trading_agent.store.models import Base, RiskState


@pytest.fixture(autouse=True)
def fresh_db() -> None:
    """Each test gets a clean in-memory database (shared static pool)."""
    db.init_engine("sqlite:///:memory:")
    yield


def seed_risk_state(settings: Settings) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with db.SessionLocal() as session:
        Base.metadata.drop_all(session.get_bind())
        Base.metadata.create_all(session.get_bind())
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


@pytest.fixture
def base_settings() -> Settings:
    return Settings(
        paper_starting_equity=10_000,
        risk_per_trade=0.01,
        max_positions=2,
        max_exposure=0.5,
        daily_loss_limit=0.03,
        max_drawdown=0.10,
        atr_stop_mult=2.0,
        take_profit_rr=2.0,
        min_confidence=0.55,
        fee_rate=0.001,
        slippage=0.0005,
        weight_technical=0.45,
        weight_regime=0.35,
        weight_sentiment=0.20,
        side_threshold=0.25,
        # Pin notifications off so tests never reach the real .env token.
        telegram_bot_token="",
        telegram_chat_id="",
    )


@pytest.fixture
def seeded(base_settings: Settings) -> Settings:
    seed_risk_state(base_settings)
    return base_settings
