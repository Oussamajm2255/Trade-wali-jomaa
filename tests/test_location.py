"""Location quality (V-MONSTER §28): four side-aware sub-scores, 0..1."""
from __future__ import annotations

import pytest

from trading_agent.fusion.location import (
    _liquidity_proximity,
    _premium_discount,
    _vwap_relation,
    _zone_support,
    compute_location_quality,
)
from trading_agent.schema.types import Side


class Snap:
    def __init__(self, **kw) -> None:
        self.price = kw.get("price", 2000.0)
        self.entry_timeframe = "15m"
        self.indicators = {"15m": kw.get("indicators", {"atr_14": 10.0})}
        self.liquidity = kw.get("liquidity", {})
        self.vwap = kw.get("vwap", {})
        self.gold_context = kw.get("gold_context", {})
        self.structure = kw.get("structure", {})


def test_liquidity_proximity_bands() -> None:
    def liq(dist_atr):
        return {"nearest_below": {"distance_atr": dist_atr}}

    assert _liquidity_proximity(Side.LONG, liq(1.0)) == 1.0  # ideal band
    assert _liquidity_proximity(Side.LONG, liq(3.0)) == 0.7
    assert _liquidity_proximity(Side.LONG, liq(6.0)) == 0.5
    assert _liquidity_proximity(Side.LONG, liq(10.0)) == 0.3  # too far
    assert _liquidity_proximity(Side.LONG, liq(0.1)) == 0.3  # inside pool
    assert _liquidity_proximity(Side.LONG, {}) == 0.5  # no data -> neutral
    # SHORT reads the pool ABOVE.
    assert _liquidity_proximity(Side.SHORT, {"nearest_above": {"distance_atr": 1.0}}) == 1.0
    assert _liquidity_proximity(Side.SHORT, {"nearest_below": {"distance_atr": 1.0}}) == 0.5


def test_vwap_relation_mirrors_sides() -> None:
    assert _vwap_relation(Side.LONG, {"state": "reclaimed"}) == 1.0
    assert _vwap_relation(Side.LONG, {"state": "rejected"}) == 0.2
    assert _vwap_relation(Side.SHORT, {"state": "rejected"}) == 1.0
    assert _vwap_relation(Side.SHORT, {"state": "reclaimed"}) == 0.2
    assert _vwap_relation(Side.LONG, {"state": "above"}) == 0.8
    assert _vwap_relation(Side.SHORT, {"state": "below"}) == 0.8
    assert _vwap_relation(Side.LONG, {}) == 0.5


def test_premium_discount_session_range() -> None:
    ctx = {"session_high": 2010.0, "session_low": 1990.0, "price": 2005.0}
    # Price above the midpoint -> premium: bad for LONG, good for SHORT.
    assert _premium_discount(Side.LONG, ctx) == pytest.approx(0.3)
    assert _premium_discount(Side.SHORT, ctx) == pytest.approx(0.7)
    # Price at the range low -> deep discount: good for LONG.
    ctx_low = {"session_high": 2010.0, "session_low": 1990.0, "price": 1990.0}
    assert _premium_discount(Side.LONG, ctx_low) == pytest.approx(0.9)
    assert _premium_discount(Side.SHORT, ctx_low) == pytest.approx(0.1)
    # Missing range -> neutral, never fabricated.
    assert _premium_discount(Side.LONG, {}) == 0.5
    assert _premium_discount(Side.LONG, {"session_high": 2010.0}) == 0.5


def test_zone_support_prefers_untested_stop_side() -> None:
    structure = {
        "fvgs": [
            {"type": "FVG_BULLISH", "zone": [1980.0, 1990.0], "status": "untested"},
            {"type": "FVG_BEARISH", "zone": [2020.0, 2030.0], "status": "untested"},
        ]
    }
    # Bullish zone below the entry shelters a LONG stop -> strong.
    assert _zone_support(Side.LONG, structure, 2000.0, 10.0) == 1.0
    # A SHORT ignores bullish zones on its stop side (above) -> neutral.
    assert _zone_support(Side.SHORT, {"fvgs": structure["fvgs"][:1]}, 2000.0, 10.0) == 0.5
    # Bearish zone above shelters a SHORT stop.
    assert _zone_support(Side.SHORT, structure, 2000.0, 10.0) == 1.0
    # A tested zone is constructive but weaker.
    tested = {"fvgs": [{"type": "FVG_BULLISH", "zone": [1980.0, 1990.0], "status": "filled"}]}
    assert _zone_support(Side.LONG, tested, 2000.0, 10.0) == 0.7
    # No zones, no ATR, no structure -> neutral.
    assert _zone_support(Side.LONG, {}, 2000.0, 10.0) == 0.5
    assert _zone_support(Side.LONG, structure, 2000.0, 0.0) == 0.5


def test_compute_location_quality_neutral_without_data() -> None:
    quality = compute_location_quality(Side.LONG, Snap())
    assert quality == pytest.approx(0.5)
    assert compute_location_quality(Side.NEUTRAL, Snap()) == 0.5


def test_compute_location_quality_uses_phase_b_blocks() -> None:
    snap = Snap(
        liquidity={"nearest_below": {"distance_atr": 1.0}},
        vwap={"state": "reclaimed"},
        gold_context={"session_high": 2010.0, "session_low": 1990.0, "price": 1995.0},
        structure={"fvgs": [{"type": "FVG_BULLISH", "zone": [1980.0, 1990.0], "status": "untested"}]},
    )
    quality = compute_location_quality(Side.LONG, snap)
    assert 0.0 <= quality <= 1.0
    # All four sub-scores strong -> well above the neutral 0.5.
    assert quality > 0.7
