"""Deterministic regime engine (spec §6): six regimes from ADX/ATR/EMA."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.data.regime import detect_regime


def frame(close: np.ndarray, high_pad: np.ndarray | float = 0.5, low_pad: np.ndarray | float = 0.5) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("1h"), periods=n, freq="1h", tz="UTC")
    high = np.asarray(high_pad) + close if np.isscalar(high_pad) else np.asarray(high_pad) + close
    low = close - np.asarray(low_pad) if np.isscalar(low_pad) else close - np.asarray(low_pad)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )


def test_trend_up() -> None:
    df = frame(np.linspace(2000.0, 2300.0, 300))
    r = detect_regime(df)
    assert r["regime"] == "trend_up"
    assert r["ema_alignment"] == "bull"
    assert 0.0 <= r["trend_strength"] <= 1.0


def test_trend_down() -> None:
    df = frame(np.linspace(2300.0, 2000.0, 300))
    r = detect_regime(df)
    assert r["regime"] == "trend_down"
    assert r["ema_alignment"] == "bear"


def test_high_volatility() -> None:
    # Flat price but ranges that expand exponentially: latest ATR dwarfs
    # its own recent median -> volatility expansion branch fires first.
    n = 300
    close = np.full(n, 2400.0)
    pad = 0.1 * np.power(1.02, np.arange(n))
    df = frame(close, high_pad=pad, low_pad=pad)
    r = detect_regime(df)
    assert r["regime"] == "high_volatility"
    assert r["atr_ratio_vs_median"] >= 1.8


def test_low_volatility() -> None:
    n = 300
    close = np.full(n, 2400.0)
    pad = 5.0 * np.power(0.98, np.arange(n))
    df = frame(close, high_pad=pad, low_pad=pad)
    r = detect_regime(df)
    assert r["regime"] == "low_volatility"
    assert r["atr_ratio_vs_median"] <= 0.55


def test_transition_when_emas_cross() -> None:
    # Long uptrend, then a gentle recent pullback: EMA20 crosses below
    # EMA50 while EMA50 stays above EMA200 -> mixed alignment = transition.
    # The pullback slope matches the base slope so ATR stays flat and the
    # volatility branch does not fire.
    close = np.linspace(1000.0, 2000.0, 300)
    close[-60:] = np.linspace(2000.0, 1800.0, 60)
    df = frame(close)
    r = detect_regime(df)
    assert r["regime"] == "transition"
    assert r["ema_alignment"] == "mixed"


def test_range_on_flat_noise() -> None:
    rng = np.random.default_rng(42)
    close = 2400.0 + rng.normal(0.0, 0.1, 300)
    df = frame(close)
    r = detect_regime(df)
    assert r["regime"] == "range"


def test_result_has_all_expected_keys() -> None:
    r = detect_regime(frame(np.linspace(2000.0, 2300.0, 300)))
    assert {"regime", "adx", "atr", "atr_ratio_vs_median", "ema_alignment", "ema20_slope", "trend_strength"} <= set(r)
