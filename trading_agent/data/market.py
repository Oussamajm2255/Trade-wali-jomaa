"""Market data access: OHLCV via ccxt (public endpoints) + Fear & Greed.

Public-only in v1 — paper trading simulates fills locally, so no private
exchange keys are ever required or accepted.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import ccxt
import pandas as pd
import requests

from trading_agent.data.indicators import build_snapshot


class MarketDataError(RuntimeError):
    """Raised when market data cannot be fetched."""


class MarketData:
    def __init__(self, exchange_id: str, timeout: int = 30) -> None:
        if not hasattr(ccxt, exchange_id):
            raise MarketDataError(f"unknown exchange id: {exchange_id!r}")
        self.exchange_id = exchange_id
        self.timeout = timeout
        self._exchange: Any = None
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def _client(self) -> Any:
        """Lazy ccxt client — avoids network I/O at import time."""
        if self._exchange is None:
            client = getattr(ccxt, self.exchange_id)(
                {"enableRateLimit": True, "timeout": self.timeout * 1000}
            )
            client.load_markets()
            self._exchange = client
        return self._exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 300,
        ttl: float = 60.0,
    ) -> pd.DataFrame:
        """Fetch OHLCV with a small in-memory cache (rate-limit friendly)."""
        key = (symbol, timeframe)
        now = time.time()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < ttl and len(cached[1]) >= limit:
                return cached[1].copy()
        try:
            raw = self._client().fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as exc:  # noqa: BLE001 - ccxt raises many exception types
            raise MarketDataError(f"fetch_ohlcv({symbol}, {timeframe}) failed: {exc}") from exc
        if not raw:
            raise MarketDataError(f"fetch_ohlcv({symbol}, {timeframe}) returned no data")
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        with self._lock:
            self._cache[key] = (now, df.copy())
        return df

    @staticmethod
    def fear_greed_index() -> dict:
        """Crypto Fear & Greed Index (alternative.me — keyless public API)."""
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
            resp.raise_for_status()
            payload = resp.json()["data"][0]
            return {
                "value": int(payload["value"]),
                "classification": str(payload["value_classification"]),
                "ts": str(payload["timestamp"]),
                "kind": "fear_greed",
            }
        except Exception as exc:  # noqa: BLE001
            raise MarketDataError(f"fear_greed_index failed: {exc}") from exc

    def analysis_input(self, symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, dict, dict | None]:
        """One call that gathers candles + indicator snapshot + sentiment.

        Fear & Greed failure is non-fatal: the pipeline degrades gracefully.
        """
        df = self.fetch_ohlcv(symbol, timeframe, limit)
        snapshot = build_snapshot(df)
        snapshot["symbol"] = symbol
        snapshot["timeframe"] = timeframe
        fng: dict | None = None
        try:
            fng = self.fear_greed_index()
        except MarketDataError:
            pass
        return df, snapshot, fng
