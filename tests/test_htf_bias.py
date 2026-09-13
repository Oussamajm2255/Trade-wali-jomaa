"""HTF bias: deterministic 4h trend classification + risk-engine hard gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.data.bias import compute_htf_bias
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import AgentVerdict, Rejection, Side

BULL = {"bias": "bull", "score": 2, "adx": 30.0, "detail": "ema50 > ema200, price > ema200, ADX 30.0"}
BEAR = {"bias": "bear", "score": -2, "adx": 28.0, "detail": "ema50 < ema200, price < ema200, ADX 28.0"}
NEUTRAL = {"bias": "neutral", "score": 0, "adx": 12.0, "detail": "ADX 12.0 < 20: no trend (chop)"}


def make_df(kind: str, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    i = np.arange(n, dtype=float)
    if kind == "bull":
        close = 1000 + 2.0 * i + rng.normal(0, 0.4, n)
    elif kind == "bear":
        close = 2000 - 2.0 * i + rng.normal(0, 0.4, n)
    else:  # chop: flat noise, no trend
        close = 1000 + rng.normal(0, 0.5, n)
    idx = pd.date_range("2026-07-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close + 0.7,
            "low": close - 0.7,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )


def test_bull_trend():
    bias = compute_htf_bias(make_df("bull"))
    assert bias["bias"] == "bull"
    assert bias["score"] == 2
    assert bias["adx"] >= 20


def test_bear_trend():
    bias = compute_htf_bias(make_df("bear"))
    assert bias["bias"] == "bear"
    assert bias["score"] == -2
    assert bias["adx"] >= 20


def test_chop_is_neutral():
    bias = compute_htf_bias(make_df("chop"))
    assert bias["bias"] == "neutral"


def test_not_enough_candles():
    bias = compute_htf_bias(make_df("bull", n=50))
    assert bias["bias"] == "neutral"
    assert bias["adx"] == 0.0


def verdicts():
    return {
        "technical": AgentVerdict(agent="technical", source="test", model="test", payload={}),
        "sentiment": AgentVerdict(agent="sentiment", source="test", model="test", payload={}),
        "regime": AgentVerdict(agent="regime", source="test", model="test", payload={}),
    }


def evaluate(settings, side, htf_bias=None):
    engine = RiskEngine(settings)
    return engine.evaluate(
        symbol="XAUUSD",
        timeframe="15m",
        side=side,
        confidence=0.8,
        price=2400.0,
        atr=5.0,
        verdicts=verdicts(),
        gauge=None,
        htf_bias=htf_bias,
    )


def test_long_requires_bull_bias(base_settings):
    ok = evaluate(base_settings, Side.LONG, BULL)
    assert not isinstance(ok, Rejection)
    rej = evaluate(base_settings, Side.LONG, BEAR)
    assert isinstance(rej, Rejection) and "HTF bias" in rej.reason


def test_short_requires_bear_bias(base_settings):
    ok = evaluate(base_settings, Side.SHORT, BEAR)
    assert not isinstance(ok, Rejection)
    rej = evaluate(base_settings, Side.SHORT, BULL)
    assert isinstance(rej, Rejection) and "HTF bias" in rej.reason


def test_neutral_bias_blocks_both_sides(base_settings):
    assert isinstance(evaluate(base_settings, Side.LONG, NEUTRAL), Rejection)
    assert isinstance(evaluate(base_settings, Side.SHORT, NEUTRAL), Rejection)


def test_gate_disabled_passes_any_bias(base_settings):
    settings = base_settings.model_copy(update={"htf_bias_filter_enabled": False})
    assert not isinstance(evaluate(settings, Side.LONG, BEAR), Rejection)
    assert not isinstance(evaluate(settings, Side.SHORT, BULL), Rejection)


def test_missing_bias_skips_gate(base_settings):
    # Backward compatible: no htf_bias provided -> gate not applied.
    assert not isinstance(evaluate(base_settings, Side.LONG, None), Rejection)
