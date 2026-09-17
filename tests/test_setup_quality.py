"""Setup Quality Engine tests (spec §18): seven transparent components,
deterministic from the canonical snapshot, never from the LLM."""
from __future__ import annotations

import pytest

from trading_agent.config import Settings
from trading_agent.fusion.setup_quality import (
    WEIGHTS,
    _session_component,
    _volatility_component,
    compute_setup_quality,
)
from trading_agent.schema.types import Side


class Snap:
    """Duck-typed MarketSnapshot stand-in (the engine only reads context)."""

    def __init__(self, **kw) -> None:
        self.price = kw.get("price", 2000.0)
        self.entry_timeframe = kw.get("entry_timeframe", "15m")
        self.indicators = {
            self.entry_timeframe: kw.get(
                "indicators", {"atr_14": 10.0, "atr_percentile_100": 0.5}
            )
        }
        self.alignment = kw.get("alignment", {})
        self.structure = kw.get("structure", {})
        self.regimes = {self.entry_timeframe: kw.get("regime", {})}
        self.dxy = kw.get("dxy")
        self.dxy_context = kw.get("dxy_context")
        self.session_context = kw.get("session_context", {})


@pytest.fixture
def settings() -> Settings:
    return Settings(atr_stop_mult=2.0, take_profit_rr=2.0)


def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_perfect_long_setup_scores_high(settings: Settings) -> None:
    snap = Snap(
        alignment={"alignment": "BULLISH_ALIGNMENT"},
        structure={"bos": [{"type": "BOS_BULLISH"}], "support": [1960.0]},
        regime={"regime": "trend_up"},
        dxy={"kind": "dxy", "value": 70},
        session_context={"label": "LONDON_NY_OVERLAP"},
        indicators={"atr_14": 10.0, "atr_percentile_100": 0.5},
    )
    quality = compute_setup_quality(Side.LONG, snap, settings)
    # mtf 1.0 | structure 0.9 | regime 1.0 | dxy 1.0 | vol 0.9 |
    # session 1.0 | rr 1.0 (room 40 / stop 20 == take_profit_rr) |
    # location 0.5 (no Phase B blocks on this snap -> neutral)
    expected = (
        0.18 * 1.0 + 0.12 * 0.9 + 0.18 * 1.0 + 0.12 * 1.0
        + 0.09 * 0.9 + 0.09 * 1.0 + 0.07 * 1.0 + 0.15 * 0.5
    )
    assert quality.score == pytest.approx(expected)
    assert quality.components["risk_reward"] == 1.0
    assert quality.detail == "no weak components"


def test_poor_contradicted_setup_scores_low(settings: Settings) -> None:
    snap = Snap(
        alignment={"alignment": "BEARISH_ALIGNMENT"},
        structure={"choch": [{"type": "CHOCH_BEARISH"}]},
        regime={"regime": "trend_down"},
        dxy={"kind": "dxy", "value": 30},
        session_context={"label": "OFF_SESSION"},
        indicators={"atr_14": 10.0, "atr_percentile_100": 0.95},
    )
    quality = compute_setup_quality(Side.LONG, snap, settings)
    assert quality.score < 0.3
    assert quality.components["regime"] == 0.2
    assert quality.components["mtf_alignment"] == 0.15
    assert "structure" in quality.detail  # weakest components are named


def test_neutral_side_is_neutral_quality(settings: Settings) -> None:
    quality = compute_setup_quality(Side.NEUTRAL, Snap(indicators={}), settings)
    assert quality.score == pytest.approx(0.5)


def test_volatility_component_band_edges() -> None:
    assert _volatility_component(0.5) == 0.9
    assert _volatility_component(0.05) == 0.35
    assert _volatility_component(0.99) == 0.35
    assert _volatility_component(None) == 0.5
    # Monotonic inside the bands: moving toward the middle never lowers.
    assert _volatility_component(0.15) < _volatility_component(0.24)
    assert _volatility_component(0.85) > _volatility_component(0.89)


def test_session_component_labels() -> None:
    assert _session_component({"label": "LONDON_NY_OVERLAP"}) == 1.0
    assert _session_component({"label": "NEW_YORK"}) == 0.8
    assert _session_component({"label": "ASIA"}) == 0.5
    assert _session_component({"label": "OFF_SESSION"}) == 0.3
    assert _session_component({}) == 0.5


def test_dxy_direct_relationship_caps_quality(settings: Settings) -> None:
    snap = Snap(
        alignment={"alignment": "BULLISH_ALIGNMENT"},
        structure={"bos": [{"type": "BOS_BULLISH"}]},
        regime={"regime": "trend_up"},
        dxy={"kind": "dxy", "value": 70},
        dxy_context={"xau_vs_dxy": {"relationship_1h": "direct"}},
        session_context={"label": "LONDON_NY_OVERLAP"},
    )
    quality = compute_setup_quality(Side.LONG, snap, settings)
    assert quality.components["dxy"] == 0.5  # 1.0 capped by the direct relationship


def test_dxy_divergence_caps_quality_harder(settings: Settings) -> None:
    snap = Snap(
        dxy={"kind": "dxy", "value": 70},
        dxy_context={"xau_vs_dxy": {"divergence": True}},
    )
    quality = compute_setup_quality(Side.LONG, snap, settings)
    assert quality.components["dxy"] == 0.35


def test_missing_context_degrades_to_neutral(settings: Settings) -> None:
    quality = compute_setup_quality(Side.LONG, Snap(indicators={}), settings)
    assert quality.components == {
        "mtf_alignment": 0.45,
        "structure": 0.5,
        "regime": 0.5,
        "dxy": 0.5,
        "volatility": 0.5,
        "session": 0.5,
        "risk_reward": 0.5,
        "location": 0.5,
    }
