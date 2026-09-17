"""Market structure & SMC detection (spec §7/§8): swings, BOS/CHoCH,
equal highs/lows, sweeps, FVGs, displacement + order-block candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.data.structure import detect_structure


def flat_frame(n: int = 60) -> pd.DataFrame:
    """Flat candles around 2400: TR ~1, no accidental swings or gaps."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=n, freq="15min", tz="UTC")
    close = np.full(n, 2400.0)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 100.0},
        index=idx,
    )


def zigzag(n: int = 130, rising: bool = True) -> pd.DataFrame:
    """10-candle legs between alternating swing levels (rising or falling)."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=n, freq="15min", tz="UTC")
    close = np.empty(n)
    for i in range(n):
        k = i // 10
        pos = (i % 10) / 10.0  # peak is a single candle (unique swing)
        if rising:
            if k % 2 == 0:  # low -> high leg
                close[i] = (2390.0 + 10 * (k // 2)) + 20 * pos
            else:  # high -> low leg
                close[i] = (2410.0 + 10 * (k // 2)) - 10 * pos
        else:
            if k % 2 == 0:  # low -> high leg (levels falling over blocks)
                close[i] = (2400.0 - 10 * (k // 2)) + 10 * pos
            else:
                close[i] = (2410.0 - 10 * (k // 2)) - 10 * pos
    return pd.DataFrame(
        {"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 100.0},
        index=idx,
    )


def test_swing_labels_and_event_fields() -> None:
    df = zigzag(120, rising=True)
    r = detect_structure(df, timeframe="15m")
    highs = [s for s in r["swings"] if s["type"] == "SWING_HIGH"]
    lows = [s for s in r["swings"] if s["type"] == "SWING_LOW"]
    assert len(highs) >= 4 and len(lows) >= 4
    assert highs[1]["label"] == "HH"  # rising highs
    assert lows[1]["label"] == "HL"  # rising lows
    for ev in r["swings"]:
        assert ev["timeframe"] == "15m"
        assert ev["candle_index"] >= 0
        assert ev["age"] >= 0
        assert ev["timestamp"]


def test_bos_after_break_of_last_swing_high() -> None:
    df = zigzag(110, rising=True)
    # Flat block above the last swing high (~2450): no new swing forms,
    # so the close break at candle 100 is the BOS.
    df.iloc[100:, df.columns.get_loc("close")] = 2470.0
    df.iloc[100:, df.columns.get_loc("open")] = 2470.0
    df.iloc[100:, df.columns.get_loc("high")] = 2470.2
    df.iloc[100:, df.columns.get_loc("low")] = 2469.8
    r = detect_structure(df)
    assert any(e["type"] == "BOS_BULLISH" for e in r["bos"])
    assert r["bos"][0]["status"] == "active"


def test_choch_against_downtrend() -> None:
    df = zigzag(110, rising=False)
    # Flat block above the last swing high (~2370) against a downtrend:
    # the close break is a CHoCH, not a continuation.
    df.iloc[100:, df.columns.get_loc("close")] = 2380.0
    df.iloc[100:, df.columns.get_loc("open")] = 2380.0
    df.iloc[100:, df.columns.get_loc("high")] = 2380.2
    df.iloc[100:, df.columns.get_loc("low")] = 2379.8
    r = detect_structure(df)
    assert any(e["type"] == "CHOCH_BULLISH" for e in r["choch"])


def test_equal_highs_cluster() -> None:
    df = flat_frame(60)
    for i in (10, 30):
        df.iloc[i, df.columns.get_loc("high")] = 2410.0
        df.iloc[i, df.columns.get_loc("low")] = 2409.5
        df.iloc[i, df.columns.get_loc("open")] = 2409.7
        df.iloc[i, df.columns.get_loc("close")] = 2409.8
    r = detect_structure(df, tolerance_pct=0.05)
    assert any(abs(level - 2410.0) < 0.01 for level in r["equal_highs"])


def test_liquidity_sweep_detected() -> None:
    df = flat_frame(60)
    for i in (10, 30):
        df.iloc[i, df.columns.get_loc("high")] = 2410.0
        df.iloc[i, df.columns.get_loc("low")] = 2409.5
        df.iloc[i, df.columns.get_loc("open")] = 2409.7
        df.iloc[i, df.columns.get_loc("close")] = 2409.8
    # Wick far above the pool, close back below: a sweep.
    df.iloc[50, df.columns.get_loc("high")] = 2412.5
    df.iloc[50, df.columns.get_loc("low")] = 2409.0
    df.iloc[50, df.columns.get_loc("open")] = 2412.0
    df.iloc[50, df.columns.get_loc("close")] = 2409.5
    r = detect_structure(df, sweep_lookback=30)
    assert any(e["type"] == "LIQUIDITY_SWEEP" for e in r["sweeps"])
    sweep = next(e for e in r["sweeps"] if e["type"] == "LIQUIDITY_SWEEP")
    assert sweep["status"] == "swept"


def test_fvg_detected_and_status() -> None:
    df = flat_frame(60)
    # Candle 40 opens above candle 38's high: a bullish gap.
    df.iloc[38, df.columns.get_loc("high")] = 2408.0
    df.iloc[40, df.columns.get_loc("low")] = 2410.0
    df.iloc[40, df.columns.get_loc("high")] = 2410.2
    df.iloc[40, df.columns.get_loc("open")] = 2410.1
    df.iloc[40, df.columns.get_loc("close")] = 2410.15
    r = detect_structure(df)
    fvg = next(e for e in r["fvgs"] if e["type"] == "FVG_BULLISH")
    assert fvg["zone"] == [2408.0, 2410.0]
    assert fvg["status"] == "untested"
    # Later price trades back into the zone -> filled.
    df.iloc[52, df.columns.get_loc("low")] = 2409.0
    df.iloc[52, df.columns.get_loc("high")] = 2411.0
    r = detect_structure(df)
    fvg = next(e for e in r["fvgs"] if e["type"] == "FVG_BULLISH")
    assert fvg["status"] == "filled"


def test_displacement_and_order_block() -> None:
    df = flat_frame(60)
    # Candle 39 a small bearish candle below the flat area, candle 40 a
    # bullish displacement (range >= 1.5 ATR) that never revisits it.
    df.iloc[39, df.columns.get_loc("open")] = 2396.2
    df.iloc[39, df.columns.get_loc("close")] = 2395.8
    df.iloc[39, df.columns.get_loc("high")] = 2396.5
    df.iloc[39, df.columns.get_loc("low")] = 2395.5
    df.iloc[40, df.columns.get_loc("open")] = 2395.0
    df.iloc[40, df.columns.get_loc("close")] = 2404.0
    df.iloc[40, df.columns.get_loc("high")] = 2404.5
    df.iloc[40, df.columns.get_loc("low")] = 2394.5
    r = detect_structure(df)
    assert any(e["type"] == "DISPLACEMENT" and e["direction"] == "bullish" for e in r["displacements"])
    ob = next(e for e in r["order_blocks"] if e["direction"] == "bullish")
    assert ob["candle_index"] == 39
    assert ob["status"] == "untested"


def test_displacement_quality_scored_with_follow_through() -> None:
    df = flat_frame(60)
    # Candle 55: a strong bullish displacement (range 5, body 4/5).
    df.iloc[55, df.columns.get_loc("open")] = 2398.0
    df.iloc[55, df.columns.get_loc("close")] = 2402.0
    df.iloc[55, df.columns.get_loc("high")] = 2402.5
    df.iloc[55, df.columns.get_loc("low")] = 2397.5
    # Candle 57 gaps above candle 55's high: same-direction FVG inside
    # the follow window.
    df.iloc[57, df.columns.get_loc("open")] = 2404.0
    df.iloc[57, df.columns.get_loc("close")] = 2404.2
    df.iloc[57, df.columns.get_loc("high")] = 2404.5
    df.iloc[57, df.columns.get_loc("low")] = 2403.5
    r = detect_structure(df)
    dq = r["displacement_quality"]
    assert dq is not None
    assert dq["direction"] == "bullish"
    assert dq["candle_index"] == 55
    # ATR also absorbs the FVG gap candle's true range, so 5/ATR lands
    # just under the 2x saturation threshold — honest, not 1.0.
    assert dq["components"]["range"] == pytest.approx(0.95, abs=0.05)
    assert dq["components"]["body"] == 1.0  # body ratio 0.8 >= 0.7
    assert dq["components"]["consecutive"] == 0.3  # single candle
    assert dq["components"]["follow_through"] == 1.0  # FVG 2 candles later
    assert dq["quality"] == pytest.approx(0.84, abs=0.02)
    # The displacement entry itself carries the score.
    disp = next(e for e in r["displacements"] if e["candle_index"] == 55)
    assert disp["quality"] == dq["quality"]


def test_displacement_quality_consecutive_candles() -> None:
    df = flat_frame(60)
    for i in (58, 59):
        df.iloc[i, df.columns.get_loc("open")] = 2398.0
        df.iloc[i, df.columns.get_loc("close")] = 2402.0
        df.iloc[i, df.columns.get_loc("high")] = 2402.5
        df.iloc[i, df.columns.get_loc("low")] = 2397.5
    r = detect_structure(df)
    dq = r["displacement_quality"]
    assert dq["candle_index"] == 59
    assert dq["components"]["consecutive"] == 0.7  # two in a row


def test_displacement_quality_none_without_displacement() -> None:
    assert detect_structure(flat_frame(60))["displacement_quality"] is None


def test_support_resistance_from_swings() -> None:
    df = zigzag(120, rising=True)
    r = detect_structure(df)
    assert r["support"] and r["resistance"]
    assert min(r["support"]) < min(r["resistance"])  # earliest levels
