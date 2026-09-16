"""Deterministic DXY context (spec §9): score keeps 100 = weak USD."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.data.dxy_context import compute_dxy_context, xau_vs_dxy


def dxy_df(n: int = 100, freq: str = "15min", slope: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor(freq), periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.arange(n) * slope
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": 10.0},
        index=idx,
    )


GAUGE = {"value": 60, "classification": "Bullish (USD weak)", "kind": "dxy"}


def test_rising_dxy_lowers_score() -> None:
    ctx = compute_dxy_context(dxy_df(slope=0.05), GAUGE)
    assert ctx["direction"] == "up"
    assert ctx["trend"] == "bull"
    assert ctx["score"] < 60  # dollar strength drags the gold-bullish score down
    assert ctx["classification"] == "Bearish (USD strong)"
    assert ctx["level"] is not None


def test_falling_dxy_raises_score() -> None:
    ctx = compute_dxy_context(dxy_df(slope=-0.05), GAUGE)
    assert ctx["direction"] == "down"
    assert ctx["trend"] == "bear"
    assert ctx["score"] > 60  # dollar weakness lifts the gold-bullish score
    assert ctx["classification"] == "Bullish (USD weak)"


def test_flat_dxy_neutral_score() -> None:
    ctx = compute_dxy_context(dxy_df(slope=0.0), {"value": 50})
    assert ctx["direction"] == "flat"
    assert ctx["trend"] == "flat"
    assert ctx["score"] == 50  # flat context keeps the neutral base
    assert ctx["classification"] == "Neutral"


def test_gauge_only_fallback_when_no_candles() -> None:
    ctx = compute_dxy_context(None, GAUGE)
    assert ctx["source"] == "gauge only"
    assert ctx["score"] == 60
    assert ctx["level"] is None
    assert ctx["classification"] == "Bullish (USD weak)"


def test_none_without_any_dxy_information() -> None:
    assert compute_dxy_context(None, None) is None
    assert compute_dxy_context(pd.DataFrame(), None) is None


def test_changes_follow_candle_granularity() -> None:
    ctx_15m = compute_dxy_context(dxy_df(freq="15min"), GAUGE)
    assert ctx_15m["change_15m_pct"] is not None
    assert ctx_15m["change_1h_pct"] is not None
    assert ctx_15m["change_4h_pct"] is not None
    ctx_1h = compute_dxy_context(dxy_df(freq="1h"), GAUGE)
    assert ctx_1h["change_15m_pct"] is None  # 1h frame cannot honestly claim 15m
    assert ctx_1h["change_1h_pct"] is not None
    assert ctx_1h["change_4h_pct"] is not None


# --- XAUUSD response / divergence (spec §14) ---


def test_xau_vs_dxy_inverse_relationship() -> None:
    gold = dxy_df(freq="15min", slope=3.0)  # gold rising
    dollar = dxy_df(freq="15min", slope=-0.3)  # dollar falling
    out = xau_vs_dxy(gold, dollar)
    assert out["gold_1h_pct"] is not None and out["gold_1h_pct"] > 0
    assert out["dxy_1h_pct"] is not None and out["dxy_1h_pct"] < 0
    assert out["relationship_1h"] == "inverse"  # typical for gold
    assert out["divergence"] is False


def test_xau_vs_dxy_direct_relationship_flags_divergence() -> None:
    gold = dxy_df(freq="15min", slope=3.0)
    dollar = dxy_df(freq="15min", slope=0.3)
    out = xau_vs_dxy(gold, dollar)
    assert out["relationship_1h"] == "direct"  # both up = unusual
    assert out["divergence"] is True


def test_xau_vs_dxy_flat_when_no_movement() -> None:
    gold = dxy_df(freq="15min", slope=0.0)
    dollar = dxy_df(freq="15min", slope=0.0)
    out = xau_vs_dxy(gold, dollar)
    assert out["relationship_1h"] == "flat"
    assert out["divergence"] is False


def test_xau_vs_dxy_none_without_frames() -> None:
    assert xau_vs_dxy(None, None) is None
    assert xau_vs_dxy(pd.DataFrame(), dxy_df()) is None
