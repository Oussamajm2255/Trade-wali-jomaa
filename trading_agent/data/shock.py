"""Market shock detection (spec §44) — deterministic, candle-based.

Abnormal candle range / ATR / volume / price movement / spread are
compared against a rolling baseline of the previous `lookback` candles
and classified:

    NORMAL              — nothing unusual
    VOLATILITY_EXPANSION — one axis >= expansion multiple (warning)
    SHOCK               — one axis >= shock multiple, extreme movement
                           or an extreme spread (blocks new entries)

The function is pure: same candles -> same verdict, so backtests stay
reproducible. Insufficient history never triggers a shock (fail open:
a young market is unknown, not dangerous).
"""

from __future__ import annotations

import pandas as pd

from trading_agent.data.indicators import atr as atr_series


class ShockState:
    NORMAL = "NORMAL"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    SHOCK = "SHOCK"


def _ratio(current: float, baseline: float) -> float:
    """current / baseline, with zero/NaN baselines treated as no signal."""
    if baseline is None or baseline <= 0 or pd.isna(baseline) or pd.isna(current):
        return 1.0
    return float(current) / float(baseline)


def detect_shock(
    df: pd.DataFrame,
    *,
    lookback: int = 60,
    shock_multiple: float = 3.0,
    expansion_multiple: float = 1.8,
    movement_pct: float = 1.0,
    spread_pct_threshold: float = 0.0,
    spread: float | None = None,
) -> dict:
    """Classify the last candle of `df` against its own recent history.

    Returns {"state", "ratios", "movement_pct", "spread_pct", "detail"}.
    """
    if len(df) < lookback + 1 or lookback < 1:
        return {
            "state": ShockState.NORMAL,
            "ratios": {},
            "movement_pct": 0.0,
            "spread_pct": None,
            "detail": "insufficient history for shock baseline",
        }
    prev = df.iloc[-lookback - 1 : -1]
    cur = df.iloc[-1]

    rng = df["high"] - df["low"]
    atr_vals = atr_series(df)
    ratios: dict[str, float] = {}
    if "volume" in df.columns:
        ratios["volume"] = _ratio(float(cur["volume"]), float(prev["volume"].median()))
    ratios["range"] = _ratio(float(rng.iloc[-1]), float(rng.iloc[-lookback - 1 : -1].median()))
    ratios["atr"] = _ratio(float(atr_vals.iloc[-1]), float(atr_vals.iloc[-lookback - 1 : -1].median()))

    close = float(cur["close"])
    body_pct = abs(float(cur["close"]) - float(cur["open"])) / close * 100.0 if close > 0 else 0.0
    prev_close = float(df.iloc[-2]["close"])
    gap_pct = abs(float(cur["open"]) - prev_close) / prev_close * 100.0 if prev_close > 0 else 0.0
    movement = max(body_pct, gap_pct)

    spread_pct = None
    if spread is not None and close > 0:
        spread_pct = float(spread) / close * 100.0

    offenders = [f"{name} {ratio:.1f}x baseline" for name, ratio in ratios.items()
                if ratio >= shock_multiple]
    if movement >= movement_pct:
        offenders.append(f"movement {movement:.2f}%")
    if spread_pct_threshold > 0 and spread_pct is not None and spread_pct >= spread_pct_threshold:
        offenders.append(f"spread {spread_pct:.2f}%")
    if offenders:
        return {
            "state": ShockState.SHOCK,
            "ratios": ratios,
            "movement_pct": round(movement, 4),
            "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
            "detail": "shock: " + "; ".join(offenders),
        }
    expanded = [f"{name} {ratio:.1f}x baseline" for name, ratio in ratios.items()
                if ratio >= expansion_multiple]
    if expanded:
        return {
            "state": ShockState.VOLATILITY_EXPANSION,
            "ratios": ratios,
            "movement_pct": round(movement, 4),
            "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
            "detail": "volatility expansion: " + "; ".join(expanded),
        }
    return {
        "state": ShockState.NORMAL,
        "ratios": ratios,
        "movement_pct": round(movement, 4),
        "spread_pct": round(spread_pct, 4) if spread_pct is not None else None,
        "detail": "normal volatility",
    }
