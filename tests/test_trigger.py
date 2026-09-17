"""Trigger quality vs trigger speed (V-MONSTER §30): separation, confirmation."""
from __future__ import annotations

import pytest

from trading_agent.fusion.trigger import compute_trigger_quality
from trading_agent.schema.types import Side


class Snap:
    def __init__(self, **kw) -> None:
        self.price = kw.get("price", 2000.0)
        self.structure = kw.get("structure", {})
        self.speed = kw.get("speed", {"state": "NORMAL"})


def _bos(etype: str, age: int = 2) -> dict:
    return {"type": etype, "age": age, "price": 2000.0}


def _sweep(level: float, age: int = 2) -> dict:
    return {"type": "LIQUIDITY_SWEEP", "age": age, "price": level}


def test_neutral_side_has_no_trigger() -> None:
    out = compute_trigger_quality(Side.NEUTRAL, Snap())
    assert out["quality"] == 0.5
    assert out["confirmed"] is False


def test_strong_long_trigger_confirmed() -> None:
    snap = Snap(
        structure={
            "bos": [_bos("BOS_BULLISH")],
            "sweeps": [_sweep(1990.0)],  # lows swept below price -> bullish
            "displacement_quality": {"quality": 1.0},
            "fvgs": [{"type": "FVG_BULLISH", "zone": [1980.0, 1990.0], "status": "untested"}],
        },
        speed={"state": "NORMAL"},
    )
    out = compute_trigger_quality(Side.LONG, snap)
    assert out["quality"] == pytest.approx(1.0)
    assert out["components"]["bos"] == 1.0
    assert out["components"]["sweep_reclaim"] == 1.0
    assert out["components"]["zone_shelter"] == 1.0
    assert out["confirmed"] is True
    assert out["speed_state"] == "NORMAL"


def test_short_trigger_mirrors_long() -> None:
    snap = Snap(
        structure={
            "bos": [_bos("BOS_BEARISH")],
            "sweeps": [_sweep(2010.0)],  # highs swept above price -> bearish
            "displacement_quality": {"quality": 1.0},
            "fvgs": [{"type": "FVG_BEARISH", "zone": [2010.0, 2020.0], "status": "untested"}],
        },
    )
    out = compute_trigger_quality(Side.SHORT, snap)
    assert out["quality"] == pytest.approx(1.0)
    assert out["confirmed"] is True


def test_extreme_speed_never_confirms() -> None:
    snap = Snap(
        structure={"bos": [_bos("BOS_BULLISH")]},
        speed={"state": "EXTREME"},
    )
    out = compute_trigger_quality(Side.LONG, snap)
    assert out["speed_state"] == "EXTREME"
    assert out["confirmed"] is False
    assert "not confirmed" in out["detail"]


def test_quality_below_confirm_min_is_not_confirmed() -> None:
    snap = Snap()  # no structure data: every axis neutral 0.5
    out = compute_trigger_quality(Side.LONG, snap)
    assert out["quality"] == pytest.approx(0.5)
    assert out["confirmed"] is False  # 0.5 < trigger_confirm_min 0.6
    # A confirm_min below the neutral score flips it confirmed.
    class S:
        trigger_confirm_min = 0.4
        trigger_event_window = 12

    out2 = compute_trigger_quality(Side.LONG, snap, S())
    assert out2["confirmed"] is True


def test_stale_events_are_ignored() -> None:
    snap = Snap(
        structure={
            "bos": [_bos("BOS_BULLISH", age=30)],  # outside the 12-candle window
            "sweeps": [],
            "fvgs": [],
        },
    )
    out = compute_trigger_quality(Side.LONG, snap)
    assert out["components"]["bos"] == 0.5  # neutral, not the stale 1.0
    assert out["components"]["sweep_reclaim"] == 0.5


def test_contradicting_bos_scores_low() -> None:
    snap = Snap(structure={"bos": [_bos("BOS_BEARISH")]})
    assert compute_trigger_quality(Side.LONG, snap)["components"]["bos"] == 0.2
    assert compute_trigger_quality(Side.SHORT, snap)["components"]["bos"] == 1.0


def test_tested_zone_is_constructive_but_weaker() -> None:
    snap = Snap(
        structure={
            "fvgs": [{"type": "FVG_BULLISH", "zone": [1980.0, 1990.0], "status": "filled"}],
        },
    )
    assert compute_trigger_quality(Side.LONG, snap)["components"]["zone_shelter"] == 0.7
