"""Deterministic tests for the indicator library."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.data.indicators import (
    adx,
    atr,
    build_snapshot,
    bollinger,
    ema,
    macd,
    rsi,
)


def make_df(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    close = np.clip(close, 1, None)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        },
        index=pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_rsi_uptrend_is_extreme() -> None:
    series = pd.Series(np.linspace(100, 200, 200))
    assert rsi(series).iloc[-1] > 95


def test_rsi_downtrend_is_extreme() -> None:
    series = pd.Series(np.linspace(200, 100, 200))
    assert rsi(series).iloc[-1] < 5


def test_ema_of_constant_is_constant() -> None:
    series = pd.Series(np.full(100, 42.0))
    result = ema(series, 20)
    assert np.allclose(result.iloc[-1], 42.0)


def test_macd_hist_is_line_minus_signal() -> None:
    df = make_df()
    result = macd(df["close"])
    assert np.allclose(result["hist"], result["macd"] - result["signal"])


def test_atr_always_positive_after_warmup() -> None:
    df = make_df()
    values = atr(df).dropna()  # Wilder's ATR needs `period` bars of warm-up
    assert not values.empty
    assert (values > 0).all()


def test_adx_in_bounds_and_finite() -> None:
    df = make_df()
    values = adx(df)
    assert np.isfinite(values).all()
    assert values.between(0, 100).all()


def test_bollinger_bands_envelope_price() -> None:
    df = make_df()
    bb = bollinger(df["close"])
    assert bb["upper"].iloc[-1] >= bb["mid"].iloc[-1] >= bb["lower"].iloc[-1]


def test_snapshot_is_json_safe() -> None:
    df = make_df()
    snap = build_snapshot(df)
    assert snap["last_close"] > 0
    assert isinstance(snap["rsi_14"], float)
    assert isinstance(snap["ema20_gt_ema50"], bool)
    # every value must be JSON-serialisable
    import json

    json.dumps(snap)
