"""Conflict detection tests (spec §19): ALIGNED / MIXED / CONFLICTED.

Every conflict axis is deterministic — a high AI confidence score can
never hide these contradictions."""
from __future__ import annotations

import pytest

from trading_agent.config import Settings
from trading_agent.fusion.conflicts import detect_conflicts
from trading_agent.fusion.types import ConflictState, NoTradeReason
from trading_agent.schema.types import Side


class Snap:
    """Duck-typed MarketSnapshot stand-in with the deterministic blocks."""

    def __init__(self, **kw) -> None:
        self.price = kw.get("price", 2000.0)
        self.entry_timeframe = "15m"
        self.biases = kw.get("biases", {})
        self.alignment = kw.get("alignment", {})
        self.regimes = {"15m": kw.get("regime", {})}
        self.dxy = kw.get("dxy")
        self.structure = kw.get("structure", {})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        htf_timeframe="4h",
        dxy_long_min=55.0,
        dxy_short_max=45.0,
        conflict_conflicted_min=2,
    )


def test_clean_long_is_aligned(settings: Settings) -> None:
    snap = Snap(
        biases={"4h": {"bias": "bull"}},
        alignment={"alignment": "BULLISH_ALIGNMENT"},
        regime={"regime": "trend_up"},
        dxy={"kind": "dxy", "value": 70},
        structure={"bos": [{"type": "BOS_BULLISH"}]},
    )
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.state == ConflictState.ALIGNED
    assert report.conflicts == []
    assert report.conflict_score == 0.0


def test_two_axes_are_conflicted(settings: Settings) -> None:
    snap = Snap(
        biases={"4h": {"bias": "bear"}},
        regime={"regime": "trend_down"},
        dxy={"kind": "dxy", "value": 40},
    )
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.state == ConflictState.CONFLICTED
    assert {c.axis for c in report.conflicts} == {"mtf", "regime", "dxy"}
    assert report.conflict_score == 1.0
    assert report.dominant_reason == NoTradeReason.MTF_CONFLICT


def test_single_axis_is_mixed(settings: Settings) -> None:
    snap = Snap(
        biases={"4h": {"bias": "bull"}},
        regime={"regime": "trend_down"},  # the only contradiction
        dxy={"kind": "dxy", "value": 70},
    )
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.state == ConflictState.MIXED
    assert len(report.conflicts) == 1
    assert report.conflicts[0].no_trade_reason == NoTradeReason.REGIME_CONFLICT
    assert report.conflict_score == 0.5


def test_conflicted_min_is_configurable(settings: Settings) -> None:
    snap = Snap(
        biases={"4h": {"bias": "bear"}},
        regime={"regime": "trend_down"},
        dxy={"kind": "dxy", "value": 70},
    )
    strict = Settings(conflict_conflicted_min=3, htf_timeframe="4h")
    report = detect_conflicts(Side.LONG, snap, strict)
    assert report.state == ConflictState.MIXED  # 2 axes < 3


def test_strong_dollar_conflicts_with_long(settings: Settings) -> None:
    snap = Snap(dxy={"kind": "dxy", "value": 44})
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.conflicts[0].axis == "dxy"
    assert report.conflicts[0].no_trade_reason == NoTradeReason.DXY_CONFLICT


def test_weak_dollar_conflicts_with_short(settings: Settings) -> None:
    snap = Snap(dxy={"kind": "dxy", "value": 56})
    report = detect_conflicts(Side.SHORT, snap, settings)
    assert report.conflicts[0].no_trade_reason == NoTradeReason.DXY_CONFLICT


def test_bearish_choch_conflicts_with_long(settings: Settings) -> None:
    snap = Snap(structure={"choch": [{"type": "CHOCH_BEARISH"}]})
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.conflicts[0].no_trade_reason == NoTradeReason.STRUCTURE_CONFLICT


def test_price_below_supports_conflicts_with_long(settings: Settings) -> None:
    snap = Snap(price=1000.0, structure={"support": [1900.0, 1950.0]})
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.conflicts[0].no_trade_reason == NoTradeReason.STRUCTURE_CONFLICT


def test_price_above_resistances_conflicts_with_short(settings: Settings) -> None:
    snap = Snap(price=3000.0, structure={"resistance": [2100.0, 2200.0]})
    report = detect_conflicts(Side.SHORT, snap, settings)
    assert report.conflicts[0].no_trade_reason == NoTradeReason.STRUCTURE_CONFLICT


def test_neutral_side_is_always_aligned(settings: Settings) -> None:
    snap = Snap(
        biases={"4h": {"bias": "bear"}},
        regime={"regime": "trend_down"},
        dxy={"kind": "dxy", "value": 40},
    )
    report = detect_conflicts(Side.NEUTRAL, snap, settings)
    assert report.state == ConflictState.ALIGNED
    assert report.conflicts == []


def test_alignment_conflicted_raises_mtf_conflict(settings: Settings) -> None:
    snap = Snap(
        biases={"4h": {"bias": "bull"}},  # HTF agrees, but overall MTF is conflicted
        alignment={"alignment": "CONFLICTED", "detail": "1 bullish vs 1 bearish timeframe(s)"},
    )
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.conflicts[0].axis == "mtf"
    assert report.conflicts[0].no_trade_reason == NoTradeReason.MTF_CONFLICT


def test_non_dxy_gauge_is_ignored(settings: Settings) -> None:
    snap = Snap(dxy={"kind": "fear_greed", "value": 20})
    report = detect_conflicts(Side.LONG, snap, settings)
    assert report.state == ConflictState.ALIGNED
