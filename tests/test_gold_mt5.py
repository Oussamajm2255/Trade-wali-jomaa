"""MT5 data bridge: broker-native XAUUSD candles first, honest fallback.

GoldData prefers the connected terminal for gold OHLCV (real prices,
spread, tick volume) and falls back to the public yfinance -> PAXG chain
when the terminal is absent, disabled or failing — with a cooldown so a
dead terminal is not hammered every tick. The `volume_basis` label
("real" / "tick" / "proxy") rides every fetch for honest downstream use
(spec §4/§7).
"""
from __future__ import annotations

import sys
import time
import types

import pandas as pd
import pytest

from trading_agent.data.gold import GoldData, GoldDataError


def make_mt5_df(n: int = 300, timeframe: str = "15m", naive: bool = False) -> pd.DataFrame:
    """`n` closed bars + the in-progress bar (broker copy_rates shape)."""
    freq = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[timeframe]
    now = pd.Timestamp.now(tz="UTC")
    start = now.floor(freq)  # forming bar opens here
    idx = pd.date_range(end=start, periods=n + 1, freq=freq, tz="UTC")
    close = 2400.0 + pd.Series(range(n + 1), dtype=float) * 0.01
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 10.0},
        index=idx,
    )
    if naive:
        df.index = df.index.tz_localize(None)
    return df


def make_yf_df(n: int = 300, timeframe: str = "15m") -> pd.DataFrame:
    """Closed-candle frame ending before the current forming bar."""
    freq = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[timeframe]
    now = pd.Timestamp.now(tz="UTC")
    end = now.floor(freq) - pd.Timedelta(freq)  # last CLOSED bar opens here
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    close = 2400.0 + pd.Series(range(n), dtype=float) * 0.01
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 10.0},
        index=idx,
    )


class FakeMt5:
    """Duck-typed MT5 delegate — GoldData only calls fetch_ohlcv/tick."""

    def __init__(self, df: pd.DataFrame | None = None, fail: bool = False) -> None:
        self.df = df
        self.fail = fail
        self.calls = 0
        self.last_limit: int | None = None

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        self.calls += 1
        self.last_limit = limit
        if self.fail:
            raise RuntimeError("terminal gone")
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        return self.df

    def tick(self, symbol: str) -> dict | None:
        return {"price": 2400.0, "spread": 0.35, "age_s": 0.5}


def make_gold(mt5: FakeMt5 | None = None, **kwargs) -> GoldData:
    return GoldData(exchange_id="binance", exchange_ids=["binance"], mt5=mt5, **kwargs)


def test_mt5_preferred_when_delegate_present(monkeypatch):
    gold = make_gold(FakeMt5(df=make_mt5_df()))
    # The public chain must never be touched while MT5 answers.
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda *a, **k: pytest.fail("yfinance called"))
    df = gold.fetch_ohlcv("XAUUSD", "15m", 300)
    assert len(df) == 300  # forming bar dropped, cache contract intact
    assert gold.last_source == "MT5 broker:XAUUSD"
    assert gold.volume_basis == "tick"
    assert gold.mt5.last_limit == 301  # +1 so the drop still leaves `limit`


def test_mt5_frame_served_from_cache(monkeypatch):
    gold = make_gold(FakeMt5(df=make_mt5_df()))
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda *a, **k: pytest.fail("yfinance called"))
    gold.fetch_ohlcv("XAUUSD", "15m", 300)
    gold.fetch_ohlcv("XAUUSD", "15m", 300)  # within ttl -> cache hit
    assert gold.mt5.calls == 1


def test_mt5_failure_falls_back_to_yfinance(monkeypatch):
    gold = make_gold(FakeMt5(fail=True))
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda self, tf, lim: make_yf_df(lim, tf))
    df = gold.fetch_ohlcv("XAUUSD", "15m", 300)
    assert len(df) == 300
    assert gold.last_source == "yfinance:GC=F"
    assert gold.volume_basis == "real"
    # Cooldown engaged: the next cycle does not hammer the dead terminal.
    assert gold._mt5_disabled_until > time.time()
    gold.fetch_ohlcv("XAUUSD", "15m", 300, ttl=0)
    assert gold.mt5.calls == 1


def test_mt5_retried_after_cooldown(monkeypatch):
    gold = make_gold(FakeMt5(fail=True), mt5_cooldown_s=0.0)
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda self, tf, lim: make_yf_df(lim, tf))
    gold.fetch_ohlcv("XAUUSD", "15m", 300)
    gold.fetch_ohlcv("XAUUSD", "15m", 300, ttl=0)
    assert gold.mt5.calls == 2  # cooldown expired -> MT5 tried again


def test_mt5_empty_frame_falls_back(monkeypatch):
    gold = make_gold(FakeMt5(df=pd.DataFrame()))
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda self, tf, lim: make_yf_df(lim, tf))
    df = gold.fetch_ohlcv("XAUUSD", "15m", 300)
    assert len(df) == 300 and gold.last_source == "yfinance:GC=F"


def test_prefer_mt5_false_skips_delegate(monkeypatch):
    gold = make_gold(FakeMt5(df=make_mt5_df()), prefer_mt5=False)
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda self, tf, lim: make_yf_df(lim, tf))
    df = gold.fetch_ohlcv("XAUUSD", "15m", 300)
    assert gold.mt5.calls == 0
    assert gold.last_source == "yfinance:GC=F"
    assert gold.volume_basis == "real"


def test_naive_mt5_index_localized_before_open_drop():
    gold = make_gold(FakeMt5(df=make_mt5_df(naive=True)))
    df = gold.fetch_ohlcv("XAUUSD", "15m", 300)
    assert len(df) == 300
    assert df.index.tz is not None


def test_mt5_only_open_bar_raises_fallthrough(monkeypatch):
    """A frame with nothing but the forming bar falls back, never errors."""
    forming_only = make_mt5_df(n=1).iloc[[-1]]  # single forming candle
    gold = make_gold(FakeMt5(df=forming_only))
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda self, tf, lim: make_yf_df(lim, tf))
    df = gold.fetch_ohlcv("XAUUSD", "15m", 300)
    assert len(df) == 300 and gold.last_source == "yfinance:GC=F"


def test_proxy_fallback_sets_proxy_basis(monkeypatch):
    """PAXG fallback carries volume_basis="proxy" for honest downstream use."""

    class FakeExchange:
        def load_markets(self) -> None:
            pass

        def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 300) -> list:
            return [[1720000000000, 100.0, 101.0, 99.0, 100.5, 42.0]]

    fake = types.SimpleNamespace(binance=lambda *a, **k: FakeExchange())
    monkeypatch.setitem(sys.modules, "ccxt", fake)
    gold = make_gold()
    monkeypatch.setattr(GoldData, "_yf_fetch", lambda *a, **k: (_ for _ in ()).throw(
        GoldDataError("yfinance down")
    ))
    df = gold.fetch_ohlcv("XAUUSD", "1h", 100)
    assert len(df) == 1
    assert "proxy" in gold.last_source
    assert gold.volume_basis == "proxy"
