"""Deterministic market structure & liquidity detection (spec §7/§8).

Swings, HH/HL/LH/LL, BOS/CHoCH, support/resistance, equal highs/lows,
liquidity sweeps, displacement candles, FVGs and order-block candidates —
all computed locally with objective rules. DeepSeek may INTERPRET this
data; it must never invent it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list[int], list[int]]:
    """Pivot highs/lows: candle i is a swing when it is the unique strict
    max/min of the window [i-left, i+right]."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    sh: list[int] = []
    sl: list[int] = []
    for i in range(left, len(df) - right):
        hi = highs[i - left : i + right + 1]
        lo = lows[i - left : i + right + 1]
        if highs[i] == hi.max() and (hi == highs[i]).sum() == 1:
            sh.append(i)
        if lows[i] == lo.min() and (lo == lows[i]).sum() == 1:
            sl.append(i)
    return sh, sl


def _event(
    etype: str,
    df: pd.DataFrame,
    i: int,
    price: float,
    label: str | None = None,
    timeframe: str = "15m",
    status: str | None = None,
) -> dict:
    ev = {
        "type": etype,
        "timeframe": timeframe,
        "price": round(float(price), 8),
        "timestamp": str(df.index[i]),
        "candle_index": int(i),
        "age": int(len(df) - 1 - i),  # candles since the event
    }
    if label:
        ev["label"] = label
    if status:
        ev["status"] = status
    return ev


def _clusters(levels: list[float], tolerance_pct: float) -> list[list[float]]:
    """Group levels within tolerance_pct of each other; keep pools of >= 2."""
    clusters: list[list[float]] = []
    for price in sorted(levels):
        for cluster in clusters:
            if abs(price - cluster[0]) / max(cluster[0], 1e-9) <= tolerance_pct / 100.0:
                cluster.append(price)
                break
        else:
            clusters.append([price])
    return [c for c in clusters if len(c) >= 2]


def detect_structure(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 3,
    tolerance_pct: float = 0.05,
    fvg_min_atr_mult: float = 0.3,
    displacement_atr_mult: float = 1.5,
    sweep_lookback: int = 30,
    timeframe: str = "15m",
) -> dict:
    """One deterministic pass over the frame: every structure/SMC feature."""
    sh, sl = swing_points(df, left, right)
    swings: list[dict] = []
    last_high: dict | None = None
    last_low: dict | None = None
    for i in sh:
        ev = _event("SWING_HIGH", df, i, df["high"].iloc[i], timeframe=timeframe)
        if last_high is not None:
            ev["label"] = "HH" if ev["price"] > last_high["price"] else "LH"
        swings.append(ev)
        last_high = ev
    for i in sl:
        ev = _event("SWING_LOW", df, i, df["low"].iloc[i], timeframe=timeframe)
        if last_low is not None:
            ev["label"] = "HL" if ev["price"] > last_low["price"] else "LL"
        swings.append(ev)
        last_low = ev
    swings.sort(key=lambda e: e["candle_index"])

    # Trend state from the most recent labelled swing (HH=up, LL=down).
    trend = 0
    for ev in swings:
        if ev["type"] == "SWING_HIGH" and ev.get("label") == "HH":
            trend = 1
        elif ev["type"] == "SWING_LOW" and ev.get("label") == "LL":
            trend = -1

    # BOS: continuation break beyond the last swing in trend direction.
    # CHoCH: break of the last swing AGAINST the trend = structure change.
    bos: list[dict] = []
    choch: list[dict] = []
    if last_high is not None:
        for i in range(last_high["candle_index"] + 1, len(df)):
            close = float(df["close"].iloc[i])
            if close > last_high["price"]:
                etype = "BOS_BULLISH" if trend >= 0 else "CHOCH_BULLISH"
                (bos if etype.startswith("BOS") else choch).append(
                    _event(etype, df, i, close, timeframe=timeframe, status="active")
                )
                break
    if last_low is not None:
        for i in range(last_low["candle_index"] + 1, len(df)):
            close = float(df["close"].iloc[i])
            if close < last_low["price"]:
                etype = "BOS_BEARISH" if trend <= 0 else "CHOCH_BEARISH"
                (bos if etype.startswith("BOS") else choch).append(
                    _event(etype, df, i, close, timeframe=timeframe, status="active")
                )
                break

    # Equal highs/lows = liquidity pools; sweeps = wick beyond then reclaim.
    eqh = _clusters([s["price"] for s in swings if s["type"] == "SWING_HIGH"], tolerance_pct)
    eql = _clusters([s["price"] for s in swings if s["type"] == "SWING_LOW"], tolerance_pct)
    sweeps: list[dict] = []
    for pool in eqh:
        level = max(pool)
        for i in range(max(0, len(df) - sweep_lookback), len(df)):
            if df["high"].iloc[i] > level and df["close"].iloc[i] < level:
                sweeps.append(_event("LIQUIDITY_SWEEP", df, i, level, timeframe=timeframe, status="swept"))
                break
    for pool in eql:
        level = min(pool)
        for i in range(max(0, len(df) - sweep_lookback), len(df)):
            if df["low"].iloc[i] < level and df["close"].iloc[i] > level:
                sweeps.append(_event("LIQUIDITY_SWEEP", df, i, level, timeframe=timeframe, status="swept"))
                break

    # FVGs: 3-candle gaps larger than a fraction of ATR.
    atr_now = float(atr_series(df))
    min_gap = fvg_min_atr_mult * atr_now if atr_now > 0 else 0.0
    fvgs: list[dict] = []
    for i in range(2, len(df)):
        gap_up = float(df["low"].iloc[i]) - float(df["high"].iloc[i - 2])
        if gap_up >= min_gap:
            zone = [round(float(df["high"].iloc[i - 2]), 8), round(float(df["low"].iloc[i]), 8)]
            status = "filled" if _zone_traded(df, i, zone) else "untested"
            fvgs.append(
                {
                    **_event("FVG_BULLISH", df, i, float(df["high"].iloc[i - 2]), timeframe=timeframe, status=status),
                    "gap": round(gap_up, 8),
                    "zone": zone,
                }
            )
        gap_dn = float(df["low"].iloc[i - 2]) - float(df["high"].iloc[i])
        if gap_dn >= min_gap:
            zone = [round(float(df["low"].iloc[i]), 8), round(float(df["low"].iloc[i - 2]), 8)]
            status = "filled" if _zone_traded(df, i, zone) else "untested"
            fvgs.append(
                {
                    **_event("FVG_BEARISH", df, i, float(df["low"].iloc[i - 2]), timeframe=timeframe, status=status),
                    "gap": round(gap_dn, 8),
                    "zone": zone,
                }
            )

    # Displacement candles + order-block candidates (objective, conservative).
    disp_threshold = displacement_atr_mult * atr_now if atr_now > 0 else np.inf
    displacements: list[dict] = []
    order_blocks: list[dict] = []
    for i in range(1, len(df)):
        rng = float(df["high"].iloc[i] - df["low"].iloc[i])
        if rng < disp_threshold:
            continue
        bullish = bool(df["close"].iloc[i] >= df["open"].iloc[i])
        direction = "bullish" if bullish else "bearish"
        displacements.append(
            {**_event("DISPLACEMENT", df, i, df["close"].iloc[i], timeframe=timeframe), "direction": direction, "range": round(rng, 8)}
        )
        for j in range(i - 1, max(0, i - 6), -1):  # last opposite candle before it
            if bool(df["close"].iloc[j] >= df["open"].iloc[j]) != bullish:
                ob_zone = [
                    round(float(df["low"].iloc[j]), 8),
                    round(float(df["high"].iloc[j]), 8),
                ]
                status = "tested" if _zone_traded(df, i, ob_zone) else "untested"
                order_blocks.append(
                    {**_event("ORDER_BLOCK", df, j, df["close"].iloc[j], timeframe=timeframe, status=status), "direction": direction}
                )
                break

    support = sorted({s["price"] for s in swings if s["type"] == "SWING_LOW"})[-3:]
    resistance = sorted({s["price"] for s in swings if s["type"] == "SWING_HIGH"})[-3:]

    return {
        "timeframe": timeframe,
        "swings": swings[-20:],
        "bos": bos,
        "choch": choch,
        "equal_highs": [round(max(c), 8) for c in eqh],
        "equal_lows": [round(min(c), 8) for c in eql],
        "sweeps": sweeps,
        "fvgs": fvgs[-5:],
        "displacements": displacements[-5:],
        "order_blocks": order_blocks[-5:],
        "support": support,
        "resistance": resistance,
    }


def _zone_traded(df: pd.DataFrame, after: int, zone: list[float]) -> bool:
    """Has price traded INTO [zone_low, zone_high] since candle `after`?"""
    low, high = zone[0], zone[1]
    window = df.iloc[after + 1 :]
    if window.empty:
        return False
    return bool(((window["low"] <= high) & (window["high"] >= low)).any())


def atr_series(df: pd.DataFrame, period: int = 14) -> float:
    """Last ATR value (self-contained — avoids an import cycle)."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1])
