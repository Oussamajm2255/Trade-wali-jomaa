"""PAXG fallback: multi-exchange chain (Railway = US IPs, Binance 451)."""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from trading_agent.data.gold import GoldData, GoldDataError

RAW = [[1720000000000, 100.0, 101.0, 99.0, 100.5, 42.0]]


class FakeExchange:
    def __init__(self, name: str, fail_load: bool = False, fail_fetch: bool = False) -> None:
        self.name = name
        self.fail_load = fail_load
        self.fail_fetch = fail_fetch
        self.loads = 0
        self.fetches = 0

    def load_markets(self) -> None:
        self.loads += 1
        if self.fail_load:
            raise RuntimeError(f"{self.name} 451 restricted location")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 300) -> list:
        self.fetches += 1
        if self.fail_fetch:
            raise RuntimeError(f"{self.name} fetch boom")
        return RAW


def install_fake_ccxt(monkeypatch, behaviors: dict) -> dict:
    """behaviors: {id: FakeExchange}. Returns the registry for assertions."""
    fake = types.SimpleNamespace(**{name: (lambda n: (lambda *a, **k: behaviors[n]))(name)
                                     for name in behaviors})
    monkeypatch.setitem(sys.modules, "ccxt", fake)
    return behaviors


def make_gold(exchange_ids) -> GoldData:
    return GoldData(exchange_id="binance", exchange_ids=exchange_ids)


def test_first_exchange_used_and_cached(monkeypatch):
    reg = install_fake_ccxt(monkeypatch, {"binance": FakeExchange("binance")})
    gold = make_gold(["binance", "kraken"])
    df = gold._paxg_fetch("1h", 100)
    assert isinstance(df, pd.DataFrame) and len(df) == 1
    assert reg["binance"].loads == 1
    gold._paxg_fetch("1h", 100)  # cached: no second load_markets
    assert reg["binance"].loads == 1
    assert gold._ccxt_exchange == "binance"


def test_451_falls_back_to_kraken(monkeypatch):
    reg = install_fake_ccxt(
        monkeypatch,
        {"binance": FakeExchange("binance", fail_load=True), "kraken": FakeExchange("kraken")},
    )
    gold = make_gold(["binance", "kraken"])
    df = gold._paxg_fetch("1h", 100)
    assert len(df) == 1
    assert gold._ccxt_exchange == "kraken"
    assert reg["binance"].loads == 1 and reg["kraken"].loads == 1


def test_all_exchanges_fail(monkeypatch):
    install_fake_ccxt(
        monkeypatch,
        {
            "binance": FakeExchange("binance", fail_load=True),
            "kraken": FakeExchange("kraken", fail_load=True),
        },
    )
    with pytest.raises(GoldDataError, match="no reachable PAXG exchange"):
        make_gold(["binance", "kraken"])._paxg_fetch("1h", 100)


def test_fetch_failure_switches_exchange(monkeypatch):
    reg = install_fake_ccxt(
        monkeypatch,
        {
            "binance": FakeExchange("binance", fail_fetch=True),
            "kraken": FakeExchange("kraken"),
        },
    )
    gold = make_gold(["binance", "kraken"])
    df = gold._paxg_fetch("1h", 100)
    assert len(df) == 1
    assert gold._ccxt_exchange == "kraken"
    assert reg["binance"].fetches == 1 and reg["kraken"].fetches == 1


def test_unknown_exchange_id_is_skipped(monkeypatch):
    reg = install_fake_ccxt(monkeypatch, {"kraken": FakeExchange("kraken")})
    gold = make_gold(["ghost", "kraken"])
    df = gold._paxg_fetch("1h", 100)
    assert len(df) == 1 and gold._ccxt_exchange == "kraken"


def test_default_single_exchange_backward_compat(monkeypatch):
    reg = install_fake_ccxt(monkeypatch, {"binance": FakeExchange("binance")})
    gold = GoldData(exchange_id="binance")  # no exchange_ids -> [exchange_id]
    assert gold.exchange_ids == ["binance"]
    assert len(gold._paxg_fetch("1h", 100)) == 1
