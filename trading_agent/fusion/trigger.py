"""Trigger quality vs trigger speed (V-MONSTER §30) — kept separate.

TRIGGER_QUALITY answers "is this a good entry trigger for the side?"
from deterministic structure events: a fresh BOS in the trade
direction, a sweep-and-reclaim of the stop-side pool, the most recent
displacement candle's quality and an untested FVG/OB shelter.

TRIGGER_SPEED is the market-speed state the trigger fired in. The two
are separated deliberately: a great trigger in EXTREME speed is NOT
confirmed — the trigger has quality, the speed decides whether acting
on it right now is sane. No data is never fabricated: missing axes
score neutral 0.5.
"""
from __future__ import annotations

from trading_agent.schema.types import Side


def _recent(events: list[dict], window: int, price: float) -> list[tuple[dict, str | None]]:
    """Events inside the recency window, each tagged with its level side."""
    out = []
    for e in events:
        if e.get("age") is None or e.get("age") > window:
            continue
        level = e.get("price")
        side_ = None
        if isinstance(level, (int, float)) and price > 0:
            side_ = "below" if level < price else "above"
        out.append((e, side_))
    return out


def _bos_support(side: Side, structure: dict, window: int) -> float:
    bullish = any(
        e["type"] == "BOS_BULLISH"
        for e, _ in _recent(structure.get("bos", []), window, 0.0)
    )
    bearish = any(
        e["type"] == "BOS_BEARISH"
        for e, _ in _recent(structure.get("bos", []), window, 0.0)
    )
    if side == Side.LONG:
        return 1.0 if bullish else 0.2 if bearish else 0.5
    return 1.0 if bearish else 0.2 if bullish else 0.5


def _sweep_reclaim(side: Side, structure: dict, window: int, price: float) -> float:
    below = any(
        level_side == "below"
        for _, level_side in _recent(structure.get("sweeps", []), window, price)
    )
    above = any(
        level_side == "above"
        for _, level_side in _recent(structure.get("sweeps", []), window, price)
    )
    if side == Side.LONG:
        # Sweep of LOWS then reclaim = stop hunt resolved in the trade
        # direction.
        return 1.0 if below else 0.3 if above else 0.5
    return 1.0 if above else 0.3 if below else 0.5


def _zone_shelter(side: Side, structure: dict, price: float) -> float:
    """Untested FVG/OB on the stop side shelters the entry."""
    best = None  # 1.0 untested > 0.7 tested > none
    for e in structure.get("fvgs", []):
        zone = e.get("zone") or []
        if len(zone) != 2:
            continue
        is_bullish = e["type"] == "FVG_BULLISH"
        on_stop_side = (side == Side.LONG and is_bullish and zone[1] < price) or (
            side == Side.SHORT and not is_bullish and zone[0] > price
        )
        if not on_stop_side:
            continue
        score = 1.0 if e.get("status") == "untested" else 0.7
        best = max(best, score) if best is not None else score
    return best if best is not None else 0.5


def compute_trigger_quality(side: Side, snapshot, settings=None) -> dict:
    """TRIGGER_QUALITY + TRIGGER_SPEED for one side (0..1 + state).

    `snapshot` is duck-typed: it only needs `structure`, `speed` and
    `price` attributes (the Phase B/C/D snapshot blocks).
    """
    confirm_min = float(getattr(settings, "trigger_confirm_min", 0.6) or 0.6)
    window = int(getattr(settings, "trigger_event_window", 12) or 12)

    if side == Side.NEUTRAL:
        return {
            "quality": 0.5,
            "speed_state": None,
            "confirmed": False,
            "components": {},
            "detail": "neutral side: no trigger",
        }

    structure = getattr(snapshot, "structure", {}) or {}
    speed = getattr(snapshot, "speed", {}) or {}
    price = float(getattr(snapshot, "price", 0.0) or 0.0)

    bos = _bos_support(side, structure, window)
    sweep = _sweep_reclaim(side, structure, window, price)
    displacement = (structure.get("displacement_quality") or {}).get("quality")
    displacement = float(displacement) if displacement is not None else 0.5
    zone = _zone_shelter(side, structure, price)

    quality = round((bos + sweep + displacement + zone) / 4.0, 4)
    speed_state = speed.get("state") if speed else None
    confirmed = quality >= confirm_min and speed_state != "EXTREME"

    return {
        "quality": quality,
        "speed_state": speed_state,
        "confirmed": bool(confirmed),
        "components": {
            "bos": round(bos, 4),
            "sweep_reclaim": round(sweep, 4),
            "displacement": round(displacement, 4),
            "zone_shelter": round(zone, 4),
        },
        "detail": (
            f"trigger {quality:.2f}, speed {speed_state or 'unknown'}"
            + (", confirmed" if confirmed else ", not confirmed")
        ),
    }
