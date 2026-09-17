"""Location quality (V-MONSTER §28): where is the entry, and is it good?

Deterministic 0..1 component built from the snapshot's own Phase B
outputs: liquidity map, VWAP anchors, session-range premium/discount
and untested FVG/order-block support. Four equally weighted sub-scores,
side-aware, each explainable. Missing data is neutral (0.5) — never
fabricated (spec §4).
"""
from __future__ import annotations

from trading_agent.schema.types import Side

# distance in ATR units from price to the nearest liquidity pool on the
# trade's stop side.
_POOL_IDEAL_NEAR = 2.0  # ideal entry band: close but not inside
_POOL_IDEAL_FAR = 4.0  # still constructive
_POOL_MIN = 0.3  # inside the pool -> sweep risk, not a quality edge


def _liquidity_proximity(side: Side, liquidity: dict) -> float:
    """Proximity of the stop-side liquidity pool (ATR units)."""
    key = "nearest_below" if side == Side.LONG else "nearest_above"
    level = (liquidity or {}).get(key)
    if not level or level.get("distance_atr") is None:
        return 0.5  # no measurable pool on the stop side
    dist = float(level["distance_atr"])
    if dist < _POOL_MIN:
        return 0.3  # inside the pool: sweep risk outweighs the stop shelter
    if dist <= _POOL_IDEAL_NEAR:
        return 1.0
    if dist <= _POOL_IDEAL_FAR:
        return 0.7
    if dist <= 8:
        return 0.5
    return 0.3  # pool too far to shelter the stop


_VWAP_LONG = {"reclaimed": 1.0, "above": 0.8, "below": 0.4, "rejected": 0.2}
_VWAP_SHORT = {"rejected": 1.0, "below": 0.8, "above": 0.4, "reclaimed": 0.2}


def _vwap_relation(side: Side, vwap: dict) -> float:
    """Price vs the session VWAP anchor, side-aware."""
    state = (vwap or {}).get("state")
    if not state:
        return 0.5
    table = _VWAP_LONG if side == Side.LONG else _VWAP_SHORT
    return table.get(state, 0.5)


def _premium_discount(side: Side, gold_context: dict) -> float:
    """Entry at discount (long) / premium (short) of the session range."""
    ctx = gold_context or {}
    high = ctx.get("session_high") or ctx.get("intraday_high")
    low = ctx.get("session_low") or ctx.get("intraday_low")
    price = ctx.get("price")
    if high is None or low is None or price is None or high <= low:
        return 0.5
    mid = (high + low) / 2
    half_range = (high - low) / 2
    if half_range <= 0:
        return 0.5
    depth = (mid - price) / half_range  # +1 = range low, -1 = range high
    depth = max(-1.0, min(1.0, depth))
    score = 0.5 + depth * 0.4
    if side == Side.SHORT:
        score = 0.5 - depth * 0.4
    return round(max(0.1, min(0.9, score)), 4)


def _zone_support(side: Side, structure: dict, price: float, atr: float) -> float:
    """Untested FVG / order block sheltering the stop side."""
    if atr <= 0 or not structure:
        return 0.5
    want_bull = side == Side.LONG
    stop_side = "below" if side == Side.LONG else "above"
    best = None  # (distance_atr, untested)
    for fvg in structure.get("fvgs", []) or []:
        zone = fvg.get("zone") or []
        if len(zone) != 2:
            continue
        # Edge closest to price: top of a zone below, bottom of a zone above.
        anchor = max(zone) if stop_side == "below" else min(zone)
        on_side = anchor < price if stop_side == "below" else anchor > price
        if not on_side:
            continue
        bullish_zone = fvg.get("type") == "FVG_BULLISH"
        if bullish_zone != want_bull:
            continue
        dist = abs(price - anchor) / atr
        untested = fvg.get("status") == "untested"
        best = _better(best, (dist, untested))
    for ob in structure.get("order_blocks", []) or []:
        ob_price = ob.get("price")
        if ob_price is None:
            continue
        on_side = ob_price < price if stop_side == "below" else ob_price > price
        if not on_side:
            continue
        bullish_ob = ob.get("direction") == "bullish"
        if bullish_ob != want_bull:
            continue
        dist = abs(price - ob_price) / atr
        untested = ob.get("status") == "untested"
        best = _better(best, (dist, untested))
    if best is None:
        return 0.5
    dist, untested = best
    if dist > 4:
        return 0.5  # too far to matter
    return 1.0 if untested else 0.7


def _better(current, candidate) -> tuple:
    """Prefer the closer zone; a tested near zone beats an untested far one."""
    if current is None:
        return candidate
    cur_dist, cur_untested = current
    cand_dist, cand_untested = candidate
    if cand_dist < cur_dist * 0.75 or (cand_untested and not cur_untested and cand_dist <= cur_dist):
        return candidate
    return current


def compute_location_quality(side: Side, snapshot, settings=None) -> float:
    """LOCATION_QUALITY 0..1 (V-MONSTER §28): mean of the four sub-scores.

    `snapshot` is a MarketSnapshot (duck-typed like the setup-quality
    engine). `settings` is accepted for signature symmetry and currently
    unused — all inputs are deterministic snapshot blocks.
    """
    if side == Side.NEUTRAL:
        return 0.5
    liquidity = getattr(snapshot, "liquidity", {}) or {}
    vwap = getattr(snapshot, "vwap", {}) or {}
    gold_context = getattr(snapshot, "gold_context", {}) or {}
    structure = getattr(snapshot, "structure", {}) or {}
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    indicators = (getattr(snapshot, "indicators", {}) or {}).get(
        getattr(snapshot, "entry_timeframe", ""), {}
    )
    atr = float(indicators.get("atr_14") or 0.0)

    subscores = {
        "liquidity_proximity": _liquidity_proximity(side, liquidity),
        "vwap_relation": _vwap_relation(side, vwap),
        "premium_discount": _premium_discount(side, gold_context),
        "zone_support": _zone_support(side, structure, price, atr),
    }
    return round(sum(subscores.values()) / 4, 4)
