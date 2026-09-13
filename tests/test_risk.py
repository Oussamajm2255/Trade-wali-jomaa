"""Risk-engine gate tests: sizing, limits, kill-switch persistence."""
from __future__ import annotations

import pytest

import trading_agent.store.db as db
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import AgentVerdict, Side, SignalProposal, Rejection
from trading_agent.store.models import Position


def verdicts() -> dict[str, AgentVerdict]:
    return {
        "technical": AgentVerdict(
            agent="technical",
            model="test",
            payload={"bias": "long", "conviction": 0.8, "support": 100, "resistance": 200, "notes": "t"},
        ),
        "regime": AgentVerdict(
            agent="regime", model="test", payload={"regime": "trending_up", "trend_strength": 0.6, "notes": "r"}
        ),
        "sentiment": AgentVerdict(
            agent="sentiment", model="test", payload={"score": 0.3, "tone": "greed", "notes": "s"}
        ),
    }


def test_long_sizing_math(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, SignalProposal)
    assert result.stop == pytest.approx(49_000.0)  # 2 * ATR
    assert result.target == pytest.approx(52_000.0)  # 2R
    # risk 1% of 10k = 100 USD over 1000 stop distance -> size 0.1
    assert result.size == pytest.approx(0.1, rel=1e-6)
    assert result.side == Side.LONG


def test_short_stop_is_above(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate("ETH/USDT", "1h", Side.SHORT, 0.8, 3_000.0, 100.0, verdicts(), None)
    assert isinstance(result, SignalProposal)
    assert result.stop > result.entry > result.target


def test_neutral_rejected(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate("BTC/USDT", "1h", Side.NEUTRAL, 0.5, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, Rejection)
    assert "neutral" in result.reason


def test_low_confidence_rejected(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.4, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, Rejection)


def test_max_positions_rejected(seeded) -> None:
    engine = RiskEngine(seeded)
    with db.SessionLocal() as session:
        for symbol in ("ETH/USDT", "SOL/USDT"):
            session.add(
                Position(
                    proposal_id="x", symbol=symbol, side="long", size=0.1,
                    entry=100.0, stop=90.0, target=120.0, status="open",
                )
            )
        session.commit()
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, Rejection)
    assert "max positions" in result.reason


def test_same_symbol_position_rejected(seeded) -> None:
    engine = RiskEngine(seeded)
    with db.SessionLocal() as session:
        session.add(
            Position(
                proposal_id="x", symbol="BTC/USDT", side="long", size=0.1,
                entry=50_000.0, stop=49_000.0, target=52_000.0, status="open",
            )
        )
        session.commit()
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, Rejection)


def test_kill_switch_blocks_and_reset_reopens(seeded) -> None:
    engine = RiskEngine(seeded)
    engine.halt("manual test halt")
    assert engine.is_halted()[0]
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, Rejection)
    engine.reset_halt()
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, SignalProposal)


def test_daily_loss_triggers_kill_switch(seeded) -> None:
    engine = RiskEngine(seeded)
    with db.SessionLocal() as session:
        engine.realize_pnl(session, -350.0)  # -3.5% of 10k
        session.commit()  # caller owns the transaction (same contract as broker)
    assert engine.is_halted()[0]
    assert "daily loss" in (engine.is_halted()[1] or "")


def test_drawdown_triggers_kill_switch(seeded) -> None:
    seeded.daily_loss_limit = 1.0  # neutralise daily limit; isolate drawdown
    engine = RiskEngine(seeded)
    with db.SessionLocal() as session:
        engine.realize_pnl(session, -1_100.0)  # equity 8900 vs peak 10000 -> -11%
        session.commit()
    halted, reason = engine.is_halted()
    assert halted
    assert "drawdown" in (reason or "")


def test_exposure_cap_shrinks_size(seeded) -> None:
    seeded.max_exposure = 0.25
    engine = RiskEngine(seeded)
    # risk-sized size would be 0.1 -> 5000 USD notional (50% of equity); cap is 2500.
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, SignalProposal)
    assert result.size == pytest.approx(0.05, rel=1e-6)


def test_dust_position_rejected(seeded) -> None:
    seeded.risk_per_trade = 0.000001  # risk amount 0.01 USD -> notional 0.50 USD
    engine = RiskEngine(seeded)
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, Rejection)
