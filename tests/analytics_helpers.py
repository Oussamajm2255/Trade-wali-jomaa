"""Shared helpers for the phase-6 analytics tests (not collected by pytest)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent.config import Settings

WARMUP = 250


def settings(**overrides) -> Settings:
    """Deterministic, offline, permissive settings for fast backtests."""
    base = dict(
        timeframe="15m",
        ohlcv_limit=WARMUP,
        snapshot_timeframes=[],
        htf_bias_filter_enabled=False,
        dxy_filter_enabled=False,
        session_filter_enabled=False,
        data_quality_min_candles=5,
        data_quality_allow_degraded=True,
        setup_quality_min=0.0,
        conflict_block_conflicted=False,
        min_confidence=0.35,
        deepseek_api_key=None,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    base.update(overrides)
    return Settings(**base)


def gold_frame(n: int = 700, drift: float = 0.05, noise: float = 0.6) -> pd.DataFrame:
    """A smooth 15m uptrend: drift + deterministic random-walk noise."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    base = 3000.0 * (1.0 + drift * np.linspace(0.0, 1.0, n))
    close = base + np.cumsum(rng.normal(0.0, noise, n))
    open_ = np.empty(n)
    open_[0] = close[0] - 2.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 3.0
    low = np.minimum(open_, close) - 3.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )


def dxy_frame(days: int = 18) -> pd.DataFrame:
    """A steadily declining 1h DXY frame (weak dollar = bullish gold)."""
    n = days * 24
    idx = pd.date_range("2025-12-18", periods=n, freq="1h", tz="UTC")
    close = np.linspace(106.0, 96.0, n)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )
