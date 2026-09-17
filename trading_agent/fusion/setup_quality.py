"""Deterministic Setup Quality Engine (spec §18).

Evaluates eight transparent components in 0..1 and returns a weighted
score plus every component, so the system can explain exactly why a
setup is good or poor. All inputs come from the canonical snapshot —
the LLM has no input here. Direction is deliberately NOT a component
(spec §17): a strong directional agreement can still be a poor trade.

Component inputs (spec §18 list mapped to the documented JSON shape):
  mtf_alignment  - deterministic multi-TF alignment vs the trade side
  structure      - market structure (BOS/CHoCH, support/resistance)
  regime         - deterministic regime label vs the trade side
  dxy            - deterministic DXY gauge + gold-vs-dollar relationship
  volatility     - ATR percentile (expanded/contracted chop hurts entries)
  session        - session classification (overlap > London/NY > Asia)
  risk_reward    - room to the structural level vs the ATR stop distance
  location       - Phase C (V-MONSTER §28): liquidity proximity, VWAP
                   relation, premium/discount, FVG/OB support

Spread and liquidity (listed in §18) feed the volatility and structure
components respectively (ATR expansion and untested levels are their
observable proxies on crypto data; the BAD_SPREAD gate in the risk
engine consumes a real spread when the provider supplies one).
"""
from __future__ import annotations

from trading_agent.config import Settings
from trading_agent.fusion.location import compute_location_quality
from trading_agent.fusion.types import SetupQuality
from trading_agent.schema.types import Side

# Weights sum to 1.0; documented and deterministic so a change is a
# deliberate, versioned product decision.
WEIGHTS = {
    "mtf_alignment": 0.18,
    "structure": 0.12,
    "regime": 0.18,
    "dxy": 0.12,
    "volatility": 0.09,
    "session": 0.09,
    "risk_reward": 0.07,
    "location": 0.15,
}

# Deterministic regime label -> quality per side. Opposing trends are
# near-zero quality; chop (transition/high vol) is poor but not zero.
_REGIME_QUALITY = {
    "trend_up": {"long": 1.0, "short": 0.2},
    "trend_down": {"long": 0.2, "short": 1.0},
    "range": {"long": 0.5, "short": 0.5},
    "transition": {"long": 0.35, "short": 0.35},
    "high_volatility": {"long": 0.3, "short": 0.3},
    "low_volatility": {"long": 0.45, "short": 0.45},
}

_SESSION_QUALITY = {
    "LONDON_NY_OVERLAP": 1.0,
    "LONDON": 0.8,
    "NEW_YORK": 0.8,
    "SYDNEY": 0.6,
    "ASIA": 0.5,
    "OFF_SESSION": 0.3,
}


def _mtf_component(side: Side, alignment: dict) -> float:
    state = (alignment or {}).get("alignment", "MIXED")
    if side == Side.NEUTRAL:
        return 0.5
    bullish = state == "BULLISH_ALIGNMENT"
    bearish = state == "BEARISH_ALIGNMENT"
    if side == Side.LONG:
        if bullish:
            return 1.0
        if bearish or state == "CONFLICTED":
            return 0.15
        return 0.45  # MIXED
    if bearish:
        return 1.0
    if bullish or state == "CONFLICTED":
        return 0.15
    return 0.45


def _structure_component(side: Side, structure: dict, price: float) -> float:
    if not structure:
        return 0.5
    choch = structure.get("choch") or []
    bos = structure.get("bos") or []
    supports = structure.get("support") or []
    resistances = structure.get("resistance") or []
    if side == Side.LONG:
        # A bearish CHoCH is an active structure-change warning; a bullish
        # BOS confirms the direction; standing on support is constructive.
        if any(e.get("type") == "CHOCH_BEARISH" for e in choch):
            return 0.2
        if any(e.get("type") == "BOS_BULLISH" for e in bos):
            return 0.9
        if supports and price >= min(supports):
            return 0.7
        return 0.4
    if side == Side.SHORT:
        if any(e.get("type") == "CHOCH_BULLISH" for e in choch):
            return 0.2
        if any(e.get("type") == "BOS_BEARISH" for e in bos):
            return 0.9
        if resistances and price <= max(resistances):
            return 0.7
        return 0.4
    return 0.5


def _regime_component(side: Side, regime_label: str | None) -> float:
    if side == Side.NEUTRAL or not regime_label:
        return 0.5
    return _REGIME_QUALITY.get(regime_label, {"long": 0.5, "short": 0.5})[
        "long" if side == Side.LONG else "short"
    ]


def _dxy_component(side: Side, gauge: dict | None, dxy_context: dict | None) -> float:
    if not gauge or gauge.get("kind") != "dxy":
        return 0.5  # no deterministic dollar context -> neutral
    value = float(gauge["value"])  # 100 = USD weak (bullish gold)
    if side == Side.LONG:
        base = 1.0 if value >= 55 else 0.65 if value >= 50 else 0.4 if value > 45 else 0.2
    elif side == Side.SHORT:
        base = 1.0 if value <= 45 else 0.65 if value <= 50 else 0.4 if value < 55 else 0.2
    else:
        base = 0.5
    # Gold-vs-dollar short-term behaviour (xau_vs_dxy): same-direction
    # divergence or an atypical direct relationship is a quality drag.
    rel = (dxy_context or {}).get("xau_vs_dxy") or {}
    if rel.get("divergence"):
        base = min(base, 0.35)
    elif rel.get("relationship_1h") == "direct":
        base = min(base, 0.5)
    return base


def _volatility_component(atr_pct: float | None) -> float:
    """ATR percentile (0..1): the middle band is the tradeable zone."""
    if atr_pct is None:
        return 0.5
    if 0.25 <= atr_pct <= 0.75:
        return 0.9
    if atr_pct <= 0.1 or atr_pct >= 0.9:
        return 0.35
    if atr_pct < 0.25:
        return round(0.35 + (atr_pct - 0.1) / 0.15 * 0.25, 4)
    return round(0.9 - (atr_pct - 0.75) / 0.15 * 0.3, 4)


def _session_component(session_context: dict | None) -> float:
    label = (session_context or {}).get("label", "")
    return _SESSION_QUALITY.get(label, 0.5)


def _risk_reward_component(
    side: Side, price: float, atr: float, structure: dict, settings: Settings
) -> float:
    """Room from entry to the nearest structural level, in ATR-stop units."""
    if side == Side.NEUTRAL or atr <= 0 or not structure:
        return 0.5
    stop_distance = atr * settings.atr_stop_mult
    if side == Side.LONG:
        supports = [s for s in (structure.get("support") or []) if s < price]
        level = max(supports) if supports else None
    else:
        resistances = [r for r in (structure.get("resistance") or []) if r > price]
        level = min(resistances) if resistances else None
    if level is None:
        return 0.5  # no structural level -> no measurable room
    rr = abs(price - level) / stop_distance
    target_rr = settings.take_profit_rr
    if rr >= target_rr:
        return 1.0
    if rr <= 1.0:
        return 0.3
    return round(0.3 + 0.7 * (rr - 1.0) / (target_rr - 1.0), 4)


def compute_setup_quality(side: Side, snapshot, settings: Settings) -> SetupQuality:
    """Score the setup quality of a fused side from the canonical snapshot.

    `snapshot` is a MarketSnapshot (duck-typed: the engine reads the
    deterministic context blocks only, so tests can pass a plain object).
    """
    price = float(getattr(snapshot, "price", 0.0))
    indicators = (getattr(snapshot, "indicators", {}) or {}).get(
        getattr(snapshot, "entry_timeframe", ""), {}
    )
    atr = float(indicators.get("atr_14") or 0.0)
    components = {
        "mtf_alignment": _mtf_component(side, getattr(snapshot, "alignment", {}) or {}),
        "structure": _structure_component(side, getattr(snapshot, "structure", {}) or {}, price),
        "regime": _regime_component(
            side,
            ((getattr(snapshot, "regimes", {}) or {}).get(
                getattr(snapshot, "entry_timeframe", "")
            ) or {}).get("regime"),
        ),
        "dxy": _dxy_component(
            side, getattr(snapshot, "dxy", None), getattr(snapshot, "dxy_context", None)
        ),
        "volatility": _volatility_component(indicators.get("atr_percentile_100")),
        "session": _session_component(getattr(snapshot, "session_context", {})),
        "risk_reward": _risk_reward_component(
            side, price, atr, getattr(snapshot, "structure", {}) or {}, settings
        ),
        "location": compute_location_quality(side, snapshot, settings),
    }
    score = round(
        sum(WEIGHTS[name] * value for name, value in components.items()), 4
    )
    weak = sorted(
        (name for name, value in components.items() if value < 0.5),
        key=lambda name: components[name],
    )
    detail = (
        f"weakest: {', '.join(weak)}" if weak else "no weak components"
    )
    return SetupQuality(score=score, components=components, detail=detail)
