"""Gold (XAUUSD) market data: COMEX futures via yfinance, PAXG fallback.

GC=F (COMEX gold futures) tracks spot XAUUSD within a small basis and
provides free, keyless OHLCV. When the futures market is closed (weekend)
or Yahoo is unavailable, we fall back to PAXG/USDT — tokenised gold that
trades 24/7 on the configured ccxt exchange. The data source is always
labelled so reports stay honest about what was analysed.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trading_agent.data.indicators import build_snapshot

logger = logging.getLogger(__name__)

GOLD_YF_TICKER = "GC=F"
DXY_TICKER = "DX-Y.NYB"
PAXG_SYMBOL = "PAXG/USDT"

# yfinance interval names + fetch windows (futures trade ~23h x 5d).
_INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "60m", "4h": "60m", "1d": "1d"}
_PERIOD_MAP = {"1m": "5d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "3mo", "4h": "3mo", "1d": "2y"}
_CANDLE_DURATION = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}
STALENESS_LIMIT = "12h"  # prefer 24/7 PAXG when futures data is this old


class GoldDataError(RuntimeError):
    """Raised when gold market data cannot be fetched."""


class GoldData:
    def __init__(
        self,
        exchange_id: str = "binance",
        mt5: Any = None,
        dxy_symbol: str = "DXY_U6",
        exchange_ids: list[str] | None = None,
    ) -> None:
        self.exchange_id = exchange_id
        # PAXG fallback tries these exchanges in order until one answers
        # (e.g. Binance blocks US datacenter IPs — Railway — so Kraken
        # takes over automatically).
        self.exchange_ids = exchange_ids or [exchange_id]
        self.mt5 = mt5  # optional MT5 broker for broker-native DXY candles
        self.dxy_symbol = dxy_symbol
        self._cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()
        self._ccxt_client = None
        self._ccxt_exchange: str | None = None
        self.last_source: str = "unknown"

    # ------------------------------------------------------------- yfinance

    def _yf_fetch(self, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - dependency check
            raise GoldDataError("yfinance is not installed") from exc
        try:
            ticker = yf.Ticker(GOLD_YF_TICKER)
            data = ticker.history(
                period=_PERIOD_MAP[timeframe],
                interval=_INTERVAL_MAP[timeframe],
                auto_adjust=True,
            )
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types
            raise GoldDataError(f"yfinance fetch failed: {exc}") from exc
        if data is None or data.empty:
            raise GoldDataError("yfinance returned no data")
        df = data.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        df.index = df.index.tz_convert("UTC")
        if timeframe == "4h":  # yfinance has no 4h interval: resample 1h
            df = df.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna()
        df = self._drop_open_candle(df, timeframe)
        if df.empty:
            raise GoldDataError("no closed candles available")
        return df.tail(limit)

    @staticmethod
    def _drop_open_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Remove the in-progress candle (it changes while we analyse)."""
        now = pd.Timestamp.now(tz="UTC")
        if len(df) and now - df.index[-1] < pd.Timedelta(_CANDLE_DURATION[timeframe]):
            return df.iloc[:-1]
        return df

    # ------------------------------------------------------------ PAXG fallback

    def _paxg_client(self, exclude: str | None = None):
        """First reachable ccxt client across the configured exchanges."""
        import ccxt

        if self._ccxt_client is not None:
            return self._ccxt_client
        errors: list[str] = []
        for exchange_id in self.exchange_ids:
            if exchange_id == exclude:
                continue
            try:
                client = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30_000})
                client.load_markets()
                self._ccxt_client = client
                self._ccxt_exchange = exchange_id
                return client
            except Exception as exc:  # noqa: BLE001 - try the next exchange
                errors.append(f"{exchange_id}: {exc}")
        raise GoldDataError("no reachable PAXG exchange: " + " | ".join(errors))

    def _paxg_fetch(self, timeframe: str, limit: int) -> pd.DataFrame:
        client = self._paxg_client()
        try:
            raw = client.fetch_ohlcv(PAXG_SYMBOL, timeframe=timeframe, limit=limit)
        except Exception as exc:  # noqa: BLE001 - exchange may be gone: rebuild once
            bad = self._ccxt_exchange
            self._ccxt_client = None
            self._ccxt_exchange = None
            try:
                raw = self._paxg_client(exclude=bad).fetch_ohlcv(
                    PAXG_SYMBOL, timeframe=timeframe, limit=limit
                )
            except Exception as exc2:  # noqa: BLE001
                raise GoldDataError(f"PAXG fallback failed: {exc2}") from exc
        if not raw:
            raise GoldDataError("PAXG fallback returned no data")
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    # ------------------------------------------------------------------- API

    def fetch_ohlcv(
        self, symbol: str = "XAUUSD", timeframe: str = "1h", limit: int = 300, ttl: float = 60.0
    ) -> pd.DataFrame:
        timeframe = "1h" if timeframe not in _INTERVAL_MAP else timeframe
        key = (symbol, timeframe)
        now = time.time()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < ttl and len(cached[1]) >= limit:
                return cached[1].copy()
        source = f"yfinance:{GOLD_YF_TICKER}"
        try:
            df = self._yf_fetch(timeframe, limit)
            age = pd.Timestamp.now(tz="UTC") - df.index[-1]
            if age > pd.Timedelta(STALENESS_LIMIT):
                # Futures market closed (weekend/holiday): prefer live PAXG.
                try:
                    df = self._paxg_fetch(timeframe, limit)
                    source = f"{PAXG_SYMBOL} proxy (futures closed)"
                except GoldDataError as exc:
                    logger.warning("PAXG fallback failed, keeping futures data: %s", exc)
        except GoldDataError:
            df = self._paxg_fetch(timeframe, limit)
            source = f"{PAXG_SYMBOL} proxy (yfinance unavailable)"
        with self._lock:
            self._cache[key] = (now, df.copy())
        self.last_source = source
        return df

    def sentiment_gauge(self) -> dict | None:
        """DXY-implied gold sentiment: gold rises when the dollar weakens.

        Prefers broker-native DXY candles from MT5 (24/7, even weekends)
        when a connected MT5 source is available; falls back to yfinance.
        """
        if self.mt5 is not None:
            try:
                df = self.mt5.fetch_ohlcv(self.dxy_symbol, "1h", 300)
                gauge = self._compute_gauge(df, f"MT5 {self.dxy_symbol}")
                if gauge is not None:
                    return gauge
            except Exception as exc:  # noqa: BLE001 - any MT5 failure degrades
                logger.warning("MT5 DXY fetch failed, falling back to yfinance: %s", exc)
        return self._gauge_from_yfinance()

    def _gauge_from_yfinance(self) -> dict | None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(DXY_TICKER)
            data = ticker.history(period="1mo", interval="1d", auto_adjust=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DXY fetch failed: %s", exc)
            return None
        if data is None or data.empty:
            return None
        daily = data["Close"].rename("close").to_frame()
        return self._compute_gauge(daily, "DXY dollar index (yfinance)")

    @staticmethod
    def _compute_gauge(df: pd.DataFrame, source: str) -> dict | None:
        """Map a 7-day DXY return onto a 0-100 gold sentiment gauge."""
        if df is None or df.empty:
            return None
        daily = df["close"].resample("1D").last().dropna()
        if len(daily) < 7:
            return None
        last, prev = float(daily.iloc[-1]), float(daily.iloc[-7])
        if prev == 0:
            return None
        r7 = (last / prev - 1) * 100
        value = int(round(max(0.0, min(100.0, 50.0 - r7 * 5.0))))
        classification = (
            "Bullish (USD weak)" if value >= 60 else "Bearish (USD strong)" if value <= 40 else "Neutral"
        )
        return {
            "value": value,
            "classification": classification,
            "source": source,
            "dxy_return_7d_pct": round(r7, 2),
            "ts": str(datetime.now(timezone.utc)),
            "kind": "dxy",
        }

    def analysis_input(self, symbol: str, timeframe: str, limit: int) -> tuple[pd.DataFrame, dict, dict | None]:
        df = self.fetch_ohlcv(symbol, timeframe, limit)
        snapshot = build_snapshot(df)
        snapshot["symbol"] = symbol
        snapshot["timeframe"] = timeframe
        snapshot["data_source"] = self.last_source
        return df, snapshot, self.sentiment_gauge()
