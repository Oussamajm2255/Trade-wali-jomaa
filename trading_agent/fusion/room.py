"""Room-to-target (V-MONSTER §29): can the target breathe?

Deterministic measure of how much space the trade has before the
opposing liquidity pool, in R units and after costs (spread +
slippage). When the room is smaller than `min_room_rr` the risk engine
refuses the proposal with INSUFFICIENT_ROOM. Honest about data: no
opposing level in the liquidity map -> unavailable, which never blocks
(spec §4: no data is not a reason to trade, but it is not a block).
"""
from __future__ import annotations

from trading_agent.schema.types import Side


def compute_room(
    side: Side,
    price: float,
    atr: float,
    liquidity: dict | None,
    spread_pct: float | None = None,
    slippage_pct: float = 0.0,
    stop_mult: float = 2.0,
    min_room_rr: float = 1.0,
) -> dict:
    """Room in R units between the entry and the opposing liquidity pool.

    For a LONG the opposing pool is `nearest_above`; for a SHORT,
    `nearest_below`. Costs (spread + slippage, as % of price) are
    subtracted from the raw distance before converting to R, so the
    verdict reflects what the trade can actually capture.
    """
    out = {
        "available": False,
        "reason": None,
        "opposing_level": None,
        "room_pct": None,
        "room_r": None,
        "min_required_r": round(min_room_rr, 4),
        "insufficient": False,
    }
    if side == Side.NEUTRAL:
        out["reason"] = "neutral side"
        return out
    if price <= 0 or atr <= 0:
        out["reason"] = "invalid price/ATR"
        return out
    key = "nearest_above" if side == Side.LONG else "nearest_below"
    level = (liquidity or {}).get(key)
    if not level or level.get("price") is None:
        out["reason"] = "no opposing liquidity level mapped"
        return out

    opposing = float(level["price"])
    raw_distance = opposing - price if side == Side.LONG else price - opposing
    if raw_distance <= 0:
        out["reason"] = "price already beyond the opposing level"
        return out

    room_pct = raw_distance / price * 100.0
    costs_pct = (spread_pct or 0.0) + slippage_pct
    costs_distance = price * costs_pct / 100.0
    net_distance = raw_distance - costs_distance
    stop_distance = atr * stop_mult
    room_r = net_distance / stop_distance if stop_distance > 0 else None

    out["available"] = True
    out["opposing_level"] = round(opposing, 8)
    out["room_pct"] = round(room_pct, 4)
    out["room_r"] = round(room_r, 4) if room_r is not None else None
    out["insufficient"] = room_r is not None and room_r < min_room_rr
    if out["insufficient"]:
        out["reason"] = (
            f"{room_r:.2f} R of room after costs ({costs_pct:.3f}% spread+slippage) "
            f"below the minimum {min_room_rr:.1f} R before the opposing "
            f"liquidity at {opposing:,.2f}"
        )
    return out
