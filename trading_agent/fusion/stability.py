"""Signal stability (V-MONSTER §32): STABLE / FRAGILE / VERY_FRAGILE.

A signal is stable when small input changes do not materially change
what we would tell the trader. Two deterministic perturbations are
tested and every axis is reported separately:

- ±1 tick on price: the deterministic setup-quality score is recomputed
  at price ± `stability_tick`. The score sensitivity is the larger of
  the two deltas — a signal that only just passed the quality floor and
  falls below it on a one-tick move is fragile by construction.
- −1 candle: the newest candle is excluded and ATR + market speed are
  recomputed from the shortened frame. Entry sensitivity = how far the
  second-to-last close sits from the current price (in ATRs); stop
  sensitivity = how far the ATR-based stop distance moves (in ATRs); a
  speed-state flip marks a signal sitting on a regime boundary.

Missing history is never fabricated: insufficient candles classify as
FRAGILE (unknown is not stable) — stability is enrichment, never a hard
gate by itself. NEUTRAL sides are not classified (no signal to perturb).
"""
from __future__ import annotations

import copy
from enum import StrEnum

from trading_agent.data.indicators import atr as atr_series
from trading_agent.data.speed import compute_market_speed
from trading_agent.fusion.setup_quality import compute_setup_quality
from trading_agent.schema.types import Side


class StabilityState(StrEnum):
    STABLE = "STABLE"
    FRAGILE = "FRAGILE"
    VERY_FRAGILE = "VERY_FRAGILE"


def _tick_sensitivity(side: Side, snapshot, settings, tick: float) -> float | None:
    """Max setup-quality delta across price ± tick (deterministic)."""
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    if price <= 0 or tick <= 0:
        return None
    try:
        base = compute_setup_quality(side, snapshot, settings).score
        up = copy.copy(snapshot)
        up.price = price + tick
        down = copy.copy(snapshot)
        down.price = price - tick
        q_up = compute_setup_quality(side, up, settings).score
        q_down = compute_setup_quality(side, down, settings).score
    except Exception:  # noqa: BLE001 - enrichment, never fatal
        return None
    return round(max(abs(q_up - base), abs(q_down - base)), 6)


def _candle_sensitivity(snapshot, settings, timeframe: str) -> dict | None:
    """Entry/stop/speed sensitivity to dropping the newest candle."""
    candles = (getattr(snapshot, "candles", {}) or {}).get(timeframe)
    if candles is None or len(candles) < 3:
        return None
    cut = candles.iloc[:-1]
    try:
        atr_now = float(atr_series(candles).iloc[-1])
        atr_cut = float(atr_series(cut).iloc[-1])
    except Exception:  # noqa: BLE001 - enrichment, never fatal
        return None
    if atr_now <= 0:
        return None
    price = float(getattr(snapshot, "price", 0.0) or 0.0)
    prev_close = float(candles["close"].iloc[-2])
    stop_mult = float(getattr(settings, "atr_stop_mult", 2.0) or 2.0)
    out = {
        "entry_atr": round(abs(price - prev_close) / atr_now, 6),
        "stop_atr": round(abs(atr_cut - atr_now) * stop_mult / atr_now, 6),
        "speed_flip": False,
    }
    speed_now = (getattr(snapshot, "speed", {}) or {}).get("state")
    if speed_now:
        try:
            cut_speed = compute_market_speed(
                cut,
                timeframe=timeframe,
                window=settings.speed_window,
                fast_mult=settings.speed_fast_mult,
                extreme_mult=settings.speed_extreme_mult,
                slow_mult=settings.speed_slow_mult,
                accel_lookback=settings.speed_accel_lookback,
                accel_extreme_mult=settings.speed_accel_extreme_mult,
            )
        except Exception:  # noqa: BLE001 - enrichment, never fatal
            cut_speed = {}
        out["speed_flip"] = bool(
            cut_speed.get("state") and cut_speed["state"] != speed_now
        )
    return out


def compute_stability(side: Side, snapshot, settings=None) -> dict:
    """STABLE / FRAGILE / VERY_FRAGILE for one side, with components.

    `snapshot` is duck-typed like the other fusion helpers: it needs
    `price`, `candles`, `speed`, `entry_timeframe` and whatever
    `compute_setup_quality` reads. `settings` may be None (defaults).
    """
    if side == Side.NEUTRAL:
        return {
            "state": None,
            "components": {},
            "detail": "neutral side: not classified",
        }

    tick = float(getattr(settings, "stability_tick", 0.01) or 0.01)
    score_sens = _tick_sensitivity(side, snapshot, settings, tick)
    candle = _candle_sensitivity(
        snapshot, settings, getattr(snapshot, "entry_timeframe", "")
    )

    if score_sens is None or candle is None:
        return {
            "state": StabilityState.FRAGILE,
            "components": {"score_tick": score_sens},
            "detail": "insufficient history: not classifiable, treated as fragile",
        }

    score_frag = float(getattr(settings, "stability_score_fragile", 0.05) or 0.05)
    score_vf = float(getattr(settings, "stability_score_very_fragile", 0.15) or 0.15)
    entry_frag = float(getattr(settings, "stability_entry_fragile_atr_mult", 0.2) or 0.2)
    entry_vf = float(getattr(settings, "stability_entry_very_fragile_atr_mult", 0.4) or 0.4)
    stop_frag = float(getattr(settings, "stability_stop_fragile_atr_mult", 0.1) or 0.1)
    stop_vf = float(getattr(settings, "stability_stop_very_fragile_atr_mult", 0.25) or 0.25)

    flip = candle["speed_flip"]
    very_fragile = (
        score_sens > score_vf
        or candle["entry_atr"] > entry_vf
        or candle["stop_atr"] > stop_vf
        or (flip and score_sens > score_frag)
    )
    fragile = (
        score_sens > score_frag
        or candle["entry_atr"] > entry_frag
        or candle["stop_atr"] > stop_frag
        or flip
    )
    state = (
        StabilityState.VERY_FRAGILE
        if very_fragile
        else StabilityState.FRAGILE
        if fragile
        else StabilityState.STABLE
    )

    detail = (
        f"stability {state.value}: score_sens {score_sens:.4f}, "
        f"entry {candle['entry_atr']:.4f} ATR, stop {candle['stop_atr']:.4f} ATR"
    )
    if flip:
        detail += ", speed state flips without newest candle"
    return {
        "state": state,
        "components": {
            "score_tick": score_sens,
            "entry_atr": candle["entry_atr"],
            "stop_atr": candle["stop_atr"],
            "speed_flip": flip,
        },
        "detail": detail,
    }
