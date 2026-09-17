"""Risk-engine gate tests: sizing, limits, kill-switch persistence."""
from __future__ import annotations

import pytest

import trading_agent.store.db as db
from trading_agent.fusion.types import (
    Conflict,
    ConflictReport,
    ConflictState,
    FusionContext,
    FusionResult,
    NoTradeReason,
    SetupQuality,
)
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
    # spec §20: classified no-trade reason.
    assert result.no_trade_reason == "LOW_CONFIDENCE"


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


# --- Phase 4 (spec §20): fusion-layer no-trade gates. ---


def _fusion_ctx(
    side: Side = Side.LONG,
    quality: float = 0.8,
    state: ConflictState = ConflictState.ALIGNED,
    conflicts: list[Conflict] | None = None,
    regime: str | None = None,
    spread_pct: float | None = None,
    calibrated: float | None = None,
) -> FusionContext:
    return FusionContext(
        fusion=FusionResult(
            side=side, direction_score=0.6, raw_confidence=0.6,
            contributions={"technical": 0.6},
        ),
        setup_quality=SetupQuality(score=quality, components={"regime": quality}),
        conflict=ConflictReport(state=state, conflicts=conflicts or [], conflict_score=0.0),
        calibrated_confidence=calibrated,
        regime=regime,
        spread_pct=spread_pct,
    )


def test_low_setup_quality_no_trade(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(quality=0.2),
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.LOW_SETUP_QUALITY.value
    assert "setup quality" in result.reason


def test_conflicted_blocks_by_default(seeded) -> None:
    engine = RiskEngine(seeded)
    conflicts = [
        Conflict(axis="mtf", no_trade_reason=NoTradeReason.MTF_CONFLICT, detail="4h bias bear opposes long"),
        Conflict(axis="regime", no_trade_reason=NoTradeReason.REGIME_CONFLICT, detail="regime trend_down opposes long"),
    ]
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(state=ConflictState.CONFLICTED, conflicts=conflicts),
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.MTF_CONFLICT.value  # dominant
    assert "mtf" in result.reason and "regime" in result.reason


def test_conflicted_block_can_be_disabled(seeded) -> None:
    seeded.conflict_block_conflicted = False
    engine = RiskEngine(seeded)
    conflicts = [
        Conflict(axis="mtf", no_trade_reason=NoTradeReason.MTF_CONFLICT, detail="x"),
        Conflict(axis="regime", no_trade_reason=NoTradeReason.REGIME_CONFLICT, detail="y"),
    ]
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(state=ConflictState.CONFLICTED, conflicts=conflicts),
    )
    assert isinstance(result, SignalProposal)


def test_high_volatility_no_trade_is_opt_in(seeded) -> None:
    engine = RiskEngine(seeded)
    ctx = _fusion_ctx(regime="high_volatility")
    # Off by default: proposal passes.
    assert isinstance(
        engine.evaluate("XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, fusion_context=ctx),
        SignalProposal,
    )
    seeded.no_trade_high_volatility = True
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, fusion_context=ctx
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.HIGH_VOLATILITY.value


def test_bad_spread_no_trade(seeded) -> None:
    seeded.no_trade_max_spread_pct = 0.1
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(spread_pct=0.25),
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.BAD_SPREAD.value


# --- Phase D (V-MONSTER §27): market-speed gate, opt-in. ---


def test_extreme_speed_no_trade_is_opt_in(seeded) -> None:
    engine = RiskEngine(seeded)
    ctx = {"speed": {"state": "EXTREME", "detail": "formation 3.4x"}}
    # Off by default: proposal passes.
    assert isinstance(
        engine.evaluate("XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None, ctx),
        SignalProposal,
    )
    seeded.no_trade_extreme_speed = True
    trail: list[dict] = []
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None, ctx, trail=trail
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.ABNORMAL_SPEED.value
    assert "extreme market speed" in result.reason
    assert {"gate": "speed", "status": "reject", "detail": "formation 3.4x"} in trail


def test_non_extreme_speed_never_blocks(seeded) -> None:
    seeded.no_trade_extreme_speed = True
    engine = RiskEngine(seeded)
    for state in ("SLOW", "NORMAL", "FAST"):
        result = engine.evaluate(
            "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
            {"speed": {"state": state, "detail": f"speed {state}"}},
        )
        assert isinstance(result, SignalProposal), state


def test_speed_trail_records_state(seeded) -> None:
    engine = RiskEngine(seeded)
    trail: list[dict] = []
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
        {"speed": {"state": "FAST", "detail": "formation 2.1x"}},
        trail=trail,
    )
    assert isinstance(result, SignalProposal)
    assert [t for t in trail if t["gate"] == "speed"] == [
        {"gate": "speed", "status": "pass", "detail": "formation 2.1x"}
    ]


def test_statistical_edge_unknown_no_trade(seeded) -> None:
    seeded.require_statistical_edge = True
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(calibrated=None),
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.STATISTICAL_EDGE_UNKNOWN.value
    # With calibration available the same gate lets the proposal through.
    ok = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(calibrated=0.65),
    )
    assert isinstance(ok, SignalProposal)


def test_proposal_carries_fusion_fields(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None,
        fusion_context=_fusion_ctx(calibrated=0.62),
    )
    assert isinstance(result, SignalProposal)
    assert result.direction_score == pytest.approx(0.6)
    assert result.raw_confidence == pytest.approx(0.8)
    assert result.calibrated_confidence == 0.62
    assert result.setup_quality and result.setup_quality["score"] == 0.8
    assert result.conflicts and result.conflicts["state"] == "ALIGNED"
    assert result.evidence["fusion"]["contributions"] == {"technical": 0.6}


def test_legacy_evaluate_without_fusion_context_still_works(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate("BTC/USDT", "1h", Side.LONG, 0.8, 50_000.0, 500.0, verdicts(), None)
    assert isinstance(result, SignalProposal)
    assert result.direction_score == 0.0
    assert result.setup_quality is None
    assert result.calibrated_confidence is None
