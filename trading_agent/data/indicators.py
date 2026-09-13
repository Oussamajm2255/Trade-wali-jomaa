"""Pure, deterministic technical indicators (pandas — no TA-Lib).

Kept LLM-free on purpose: every number fed to the model is computed here,
so signals are reproducible and unit-testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, 0-100; warm-up filled with a neutral 50."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Zero-loss streaks mean pure strength (RSI 100); zero-gain streaks mean 0.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out.fillna(50.0)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(series, fast) - ema(series, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX (trend strength, 0-100, direction-agnostic)."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    atr_smooth = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().fillna(0.0)


def bollinger(series: pd.Series, period: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * std, "lower": mid - k * std})


def atr_percentile(df: pd.DataFrame, period: int = 14, lookback: int = 100) -> float:
    """Where the latest ATR sits vs the last `lookback` candles (0-1)."""
    values = atr(df, period).dropna()
    if values.empty:
        return 0.5
    return float((values.iloc[-1] >= values.tail(lookback)).mean())


def build_snapshot(df: pd.DataFrame) -> dict:
    """Latest values, JSON-safe and rounded — the only thing the LLM sees."""
    close = df["close"]
    rsi_now = rsi(close)
    macd_now = macd(close)
    atr_now = atr(df)
    adx_now = adx(df)
    bb = bollinger(close)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    last = df.iloc[-1]
    return {
        "symbol": None,  # filled by caller
        "timeframe": None,  # filled by caller
        "last_close": round(float(last["close"]), 8),
        "last_candle_ts": str(df.index[-1]),
        "rsi_14": round(float(rsi_now.iloc[-1]), 2),
        "macd_line": round(float(macd_now["macd"].iloc[-1]), 8),
        "macd_signal": round(float(macd_now["signal"].iloc[-1]), 8),
        "macd_hist": round(float(macd_now["hist"].iloc[-1]), 8),
        "atr_14": round(float(atr_now.iloc[-1]), 8),
        "atr_percentile_100": round(atr_percentile(df), 2),
        "adx_14": round(float(adx_now.iloc[-1]), 2),
        "bb_upper": round(float(bb["upper"].iloc[-1]), 8),
        "bb_mid": round(float(bb["mid"].iloc[-1]), 8),
        "bb_lower": round(float(bb["lower"].iloc[-1]), 8),
        "ema20": round(float(ema20.iloc[-1]), 8),
        "ema50": round(float(ema50.iloc[-1]), 8),
        "ema200": round(float(ema200.iloc[-1]), 8),
        "ema20_gt_ema50": bool(ema20.iloc[-1] > ema50.iloc[-1]),
        "ema50_gt_ema200": bool(ema50.iloc[-1] > ema200.iloc[-1]),
        "return_24h_pct": round(float(close.pct_change(24).iloc[-1] * 100), 2) if len(close) > 24 else None,
        "return_7d_pct": round(float(close.pct_change(168).iloc[-1] * 100), 2) if len(close) > 168 else None,
        "volume_last": round(float(last["volume"]), 2),
        "volume_ma20": round(float(df["volume"].rolling(20).mean().iloc[-1]), 2),
        "volume_ratio": round(float(last["volume"]) / float(df["volume"].rolling(20).mean().iloc[-1]), 2),
    }
