"""Room-to-target (V-MONSTER §29): compute_room math + risk-engine gate."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from trading_agent.fusion.room import compute_room
from trading_agent.fusion.types import NoTradeReason
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import AgentVerdict, Rejection, Side, SignalProposal


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


# --- compute_room unit tests. ---


def test_long_reads_nearest_above_short_reads_nearest_below() -> None:
    liquidity = {
        "nearest_above": {"price": 2020.0},
        "nearest_below": {"price": 1990.0},
    }
    long = compute_room(Side.LONG, 2000.0, 10.0, liquidity)
    assert long["available"] is True
    assert long["opposing_level"] == 2020.0
    assert long["room_pct"] == pytest.approx(1.0)  # 20 / 2000 * 100
    # No costs: 20 net distance / 20 stop = exactly 1.0 R -> not below min.
    assert long["room_r"] == pytest.approx(1.0)
    assert long["insufficient"] is False

    short = compute_room(Side.SHORT, 2000.0, 10.0, liquidity)
    assert short["available"] is True
    assert short["opposing_level"] == 1990.0
    assert short["room_pct"] == pytest.approx(0.5)
    assert short["room_r"] == pytest.approx(0.5)  # 10 / 20 -> below 1 R
    assert short["insufficient"] is True


def test_costs_shrink_room_before_r_conversion() -> None:
    room = compute_room(
        Side.LONG, 2000.0, 10.0, {"nearest_above": {"price": 2020.0}},
        spread_pct=0.01, slippage_pct=0.05,
    )
    # Costs = 0.06% of 2000 = 1.2 -> net 18.8 -> 0.94 R -> insufficient.
    assert room["room_r"] == pytest.approx(0.94)
    assert room["insufficient"] is True
    assert "spread+slippage" in room["reason"]


def test_boundary_r_crosses_with_spread() -> None:
    # Slippage-only cost (0.05% of 2000 = 1.0) leaves exactly 1.0 R -> pass.
    ok = compute_room(
        Side.LONG, 2000.0, 10.0, {"nearest_above": {"price": 2021.0}},
        slippage_pct=0.05,
    )
    assert ok["room_r"] == pytest.approx(1.0)
    assert ok["insufficient"] is False
    # Adding spread drags the same level below 1 R -> reject.
    tight = compute_room(
        Side.LONG, 2000.0, 10.0, {"nearest_above": {"price": 2021.0}},
        spread_pct=0.01, slippage_pct=0.05,
    )
    assert tight["room_r"] == pytest.approx(0.99)
    assert tight["insufficient"] is True


def test_honest_without_data_never_blocks() -> None:
    for liquidity in (
        {},
        None,
        {"nearest_above": {}},
        {"nearest_above": {"price": None}},
    ):
        room = compute_room(Side.LONG, 2000.0, 10.0, liquidity)
        assert room["available"] is False
        assert room["insufficient"] is False  # spec §4: no data is not a block
        assert room["reason"]


def test_neutral_and_invalid_inputs_unavailable() -> None:
    liquidity = {"nearest_above": {"price": 2020.0}}
    assert compute_room(Side.NEUTRAL, 2000.0, 10.0, liquidity)["available"] is False
    assert compute_room(Side.LONG, 0.0, 10.0, liquidity)["available"] is False
    assert compute_room(Side.LONG, 2000.0, 0.0, liquidity)["available"] is False


def test_price_beyond_level_is_not_available() -> None:
    room = compute_room(Side.LONG, 2000.0, 10.0, {"nearest_above": {"price": 1990.0}})
    assert room["available"] is False
    assert room["insufficient"] is False
    assert "beyond" in room["reason"]


def test_min_room_rr_threshold_configurable() -> None:
    liquidity = {"nearest_above": {"price": 2040.0}}
    at_min = compute_room(Side.LONG, 2000.0, 10.0, liquidity, min_room_rr=2.0)
    assert at_min["room_r"] == pytest.approx(2.0)
    assert at_min["insufficient"] is False  # equality is not below
    below = compute_room(
        Side.LONG, 2000.0, 10.0, {"nearest_above": {"price": 2030.0}},
        min_room_rr=2.0,
    )
    assert below["room_r"] == pytest.approx(1.5)
    assert below["insufficient"] is True


# --- Risk-engine gate (spec §29 wiring). ---


def _room_context(above: float | None = None, below: float | None = None) -> dict:
    liquidity: dict = {}
    if above is not None:
        liquidity["nearest_above"] = {"price": above}
    if below is not None:
        liquidity["nearest_below"] = {"price": below}
    return {"liquidity": liquidity}


def test_evaluate_rejects_insufficient_room(seeded) -> None:
    engine = RiskEngine(seeded)
    # LONG at 2000, stop 20; opposing pool 5 ticks away -> ~0.2 R after costs.
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
        _room_context(above=2005.0),
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.INSUFFICIENT_ROOM.value
    assert "insufficient room" in result.reason


def test_evaluate_shorts_reads_below(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.SHORT, 0.8, 2000.0, 10.0, verdicts(), None, None,
        _room_context(below=1995.0),
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.INSUFFICIENT_ROOM.value


def test_evaluate_passes_with_ample_room(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
        _room_context(above=2100.0),
    )
    assert isinstance(result, SignalProposal)


def test_evaluate_room_gate_never_blocks_without_data(seeded) -> None:
    engine = RiskEngine(seeded)
    assert isinstance(
        engine.evaluate("XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None),
        SignalProposal,
    )
    # Liquidity map present but no opposing level mapped -> still passes.
    assert isinstance(
        engine.evaluate(
            "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
            _room_context(above=None),
        ),
        SignalProposal,
    )


def test_room_gate_disabled_by_settings(seeded) -> None:
    seeded.room_gate_enabled = False
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
        _room_context(above=2005.0),
    )
    assert isinstance(result, SignalProposal)


def test_room_min_rr_respected(seeded) -> None:
    seeded.room_min_rr = 0.1
    engine = RiskEngine(seeded)
    # ~0.2 R of room passes when the minimum is lowered to 0.1 R.
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, verdicts(), None, None,
        _room_context(above=2005.0),
    )
    assert isinstance(result, SignalProposal)


def test_room_gate_consumes_fusion_spread(seeded) -> None:
    engine = RiskEngine(seeded)
    fusion = SimpleNamespace(spread_pct=0.02)
    rejection = engine._room_gate(
        "XAUUSD", Side.LONG, 2000.0, 10.0, _room_context(above=2021.0), fusion
    )
    assert isinstance(rejection, Rejection)
    assert rejection.no_trade_reason == NoTradeReason.INSUFFICIENT_ROOM.value
    # Without the spread the same level left exactly 1.0 R -> pass.
    assert engine._room_gate(
        "XAUUSD", Side.LONG, 2000.0, 10.0, _room_context(above=2021.0), None
    ) is None
