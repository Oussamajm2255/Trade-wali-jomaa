"""Signal timing, actionability and execution model (V-MONSTER §42-§49, §64).

TIMING_QUALITY answers "can the human still act on this signal?" from
five deterministic components:

- trigger maturity: how fresh the structure events the trigger fired on
  are (a brand-new BOS has its full lifetime ahead; a stale one has not),
- speed: the market-speed state the signal fired in (SLOW is actionable,
  EXTREME is a chase),
- remaining room: room-to-target in R units before the opposing
  liquidity pool,
- drift: how far price is expected to move during the human reaction
  window (reaction time + Telegram latency + spread cost) relative to
  the stop distance,
- lifecycle: how much of the opportunity TTL has already been consumed.

SIGNAL_LEAD_TIME is the expected travel time from entry to target at
the current speed-adjusted pace. ACTIONABILITY_DEADLINE is now + lead
time. EXPECTED_EXECUTION_PRICE/DRIFT project where the price is
expected to sit when the human finally acts. `too_late` fires when the
lead time is shorter than the reaction window — the orchestrator's
pre-send TOO_LATE gate consumes it. Missing data is never fabricated:
unavailable axes score a neutral 0.5 and an uncomputable pace disables
`too_late` (fails open, spec §4). The function is pure for a given
`now`, so backtests stay reproducible.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from trading_agent.data.speed import SpeedState
from trading_agent.fusion.room import compute_room
from trading_agent.schema.types import Side, utcnow

_SPEED_PACE_MULT = {
    SpeedState.SLOW: 0.5,
    SpeedState.NORMAL: 1.0,
    SpeedState.FAST: 1.5,
    SpeedState.EXTREME: 2.0,
}

_SPEED_QUALITY = {
    SpeedState.SLOW: 1.0,
    SpeedState.NORMAL: 0.8,
    SpeedState.FAST: 0.5,
    SpeedState.EXTREME: 0.2,
}

_TIMEFRAME_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _side_events(side: Side, structure: dict, price: float) -> list[dict]:
    """Structure events supporting the side, each bearing an age.

    Mirrors the trigger layer's side selection: BOS in the trade
    direction, sweeps of the stop-side pool, same-direction
    displacement candles.
    """
    events: list[dict] = []
    bos_type = "BOS_BULLISH" if side == Side.LONG else "BOS_BEARISH"
    events += [e for e in structure.get("bos", []) if e.get("type") == bos_type]
    for e in structure.get("sweeps", []):
        level = e.get("price")
        if not isinstance(level, (int, float)) or price <= 0:
            continue
        if (side == Side.LONG and level < price) or (side == Side.SHORT and level > price):
            events.append(e)
    direction = "bullish" if side == Side.LONG else "bearish"
    events += [
        e for e in structure.get("displacements", []) if e.get("direction") == direction
    ]
    return events


def _trigger_maturity(side: Side, snapshot, window: int) -> float:
    """1.0 for a trigger that just fired, 0.0 for one at its stale edge."""
    structure = getattr(snapshot, "structure", {}) or {}
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    ages = [
        float(e["age"])
        for e in _side_events(side, structure, price)
        if isinstance(e.get("age"), (int, float))
    ]
    if not ages or window <= 0:
        return 0.5  # no age data: honest neutral, never fabricated
    return _clamp(1.0 - min(ages) / float(window))


def _speed_component(speed: dict) -> float:
    state = (speed or {}).get("state")
    return _SPEED_QUALITY.get(state, 0.5)


def _pace_per_minute(speed: dict, atr: float, timeframe: str) -> float | None:
    """Speed-adjusted expected price travel per minute, None when unknown."""
    base = (speed or {}).get("range_per_minute") or (speed or {}).get("atr_per_minute")
    if base is None and atr > 0:
        minutes = _TIMEFRAME_MINUTES.get(timeframe)
        if minutes:
            base = atr / float(minutes)
    if base is None or base <= 0:
        return None
    mult = _SPEED_PACE_MULT.get((speed or {}).get("state"), 1.0)
    return float(base) * mult


def compute_timing(
    side: Side,
    snapshot,
    settings=None,
    *,
    entry: float,
    stop: float,
    target: float,
    atr: float,
    spread_pct: float | None = None,
    opportunity_age_min: float | None = None,
    telegram_latency_s: float | None = None,
    now: datetime | None = None,
) -> dict:
    """TIMING_QUALITY, lead time, deadline and expected execution drift.

    `snapshot` is duck-typed: it needs `structure`, `speed`, `liquidity`,
    `price` and `entry_timeframe` (the Phase C/D snapshot blocks).
    `now` anchors the deadline; None means wall-clock time.
    """
    now = now or utcnow()
    window = int(getattr(settings, "trigger_event_window", 12) or 12)
    ttl_min = float(getattr(settings, "opportunity_ttl_minutes", 720) or 720)
    min_rr = float(getattr(settings, "room_min_rr", 1.0) or 1.0)
    reaction_s = float(getattr(settings, "user_reaction_seconds", 120.0) or 120.0)
    if telegram_latency_s is None:
        telegram_latency_s = float(getattr(settings, "telegram_latency_s", 3.0) or 3.0)
    reaction_s += float(telegram_latency_s)
    chase_mult = max(0.0, float(getattr(settings, "max_chase_atr_mult", 0.5) or 0.0))

    speed = getattr(snapshot, "speed", {}) or {}
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    timeframe = getattr(snapshot, "entry_timeframe", "") or ""

    if side == Side.NEUTRAL:
        return {
            "quality": 0.5,
            "components": {},
            "speed_state": speed.get("state"),
            "pace_per_minute": None,
            "lead_time_s": None,
            "reaction_s": round(reaction_s, 2),
            "deadline_epoch": None,
            "deadline_iso": None,
            "expected_price": round(entry, 8),
            "expected_drift": 0.0,
            "max_chase": round(chase_mult * atr, 8) if atr > 0 else 0.0,
            "execution_zone": [round(entry, 8), round(entry, 8)],
            "too_late": False,
            "detail": "neutral side: not timed",
        }

    trigger_maturity = _trigger_maturity(side, snapshot, window)
    speed_q = _speed_component(speed)

    room = compute_room(
        side,
        entry if entry > 0 else price,
        atr,
        getattr(snapshot, "liquidity", {}) or {},
        spread_pct=spread_pct,
        stop_mult=float(getattr(settings, "atr_stop_mult", 2.0) or 2.0),
        min_room_rr=min_rr,
    )
    room_r = room.get("room_r")
    room_q = _clamp(room_r / (2.0 * min_rr)) if isinstance(room_r, (int, float)) else 0.5

    # Pace feeds drift and lead time; it is not a component of its own.
    pace = _pace_per_minute(speed, atr, timeframe)
    stop_distance = abs(entry - stop)
    reaction_min = reaction_s / 60.0
    move_during_reaction = pace * reaction_min if pace is not None else 0.0
    spread_cost = price * (spread_pct or 0.0) / 100.0
    expected_drift = move_during_reaction + spread_cost
    drift_q = (
        _clamp(1.0 - expected_drift / stop_distance) if stop_distance > 0 else 0.5
    )

    age_min = opportunity_age_min
    lifecycle_q = _clamp(1.0 - age_min / ttl_min) if isinstance(age_min, (int, float)) else 0.5

    components = {
        "trigger_maturity": round(trigger_maturity, 4),
        "speed": round(speed_q, 4),
        "remaining_room": round(room_q, 4),
        "drift": round(drift_q, 4),
        "lifecycle": round(lifecycle_q, 4),
    }
    quality = round(sum(components.values()) / float(len(components)), 4)

    lead = None
    target_distance = abs(target - entry)
    if pace is not None and pace > 0 and target_distance > 0:
        lead = min(ttl_min * 60.0, target_distance / pace * 60.0)
    deadline_dt = now + timedelta(seconds=lead) if lead is not None else None

    sign = 1.0 if side == Side.LONG else -1.0
    expected_price = entry + sign * expected_drift
    max_chase = chase_mult * atr if atr > 0 else 0.0
    if side == Side.LONG:
        execution_zone = [round(entry, 8), round(entry + max_chase, 8)]
    else:
        execution_zone = [round(entry - max_chase, 8), round(entry, 8)]

    too_late = lead is not None and lead <= reaction_s
    detail = (
        f"timing {quality:.2f}: trigger {trigger_maturity:.2f}, speed "
        f"{speed.get('state') or 'unknown'}, room {room_r if isinstance(room_r, (int, float)) else 'n/a'} R, "
        f"drift {drift_q:.2f}, lifecycle {lifecycle_q:.2f}"
    )
    if lead is not None:
        detail += f" | lead {lead:.0f}s vs reaction {reaction_s:.0f}s"
        if too_late:
            detail += " — too late"
    return {
        "quality": quality,
        "components": components,
        "speed_state": speed.get("state"),
        "pace_per_minute": round(pace, 6) if pace is not None else None,
        "lead_time_s": round(lead, 2) if lead is not None else None,
        "reaction_s": round(reaction_s, 2),
        "deadline_epoch": deadline_dt.timestamp() if deadline_dt else None,
        "deadline_iso": (
            deadline_dt.strftime("%Y-%m-%d %H:%M:%S UTC") if deadline_dt else None
        ),
        "expected_price": round(expected_price, 8),
        "expected_drift": round(expected_drift, 8),
        "max_chase": round(max_chase, 8),
        "execution_zone": execution_zone,
        "too_late": bool(too_late),
        "detail": detail,
    }
