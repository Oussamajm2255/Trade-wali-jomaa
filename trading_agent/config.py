"""Centralised, environment-driven configuration.

All secrets come from environment variables / .env — never hard-coded,
never logged. Field names map 1:1 to .env keys (case-insensitive).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- DeepSeek LLM ---
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_timeout_seconds: float = 90.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2

    # --- Market data (public endpoints only) ---
    exchange_id: str = "binance"
    symbols: str = "BTC/USDT,ETH/USDT,SOL/USDT"
    timeframe: str = "1h"
    ohlcv_limit: int = 300

    # --- Storage ---
    database_url: str = "sqlite:///trading.db"

    # --- Execution ---
    # "paper" = simulated fills on real market data (default, safe)
    # "mt5"   = REAL orders via MetaTrader 5 (IC Markets, FTMO, ...)
    execution_mode: str = "paper"
    # mt5 mode refuses to start unless this is explicitly true.
    live_acknowledged: bool = False
    mt5_login: int | None = None
    mt5_password: SecretStr | None = None
    mt5_server: str = "ICMarketsSC-Demo"
    mt5_path: str = ""  # optional; terminal64.exe is auto-detected when empty
    mt5_magic: int = 770313
    mt5_deviation_points: int = 20
    mt5_dxy_symbol: str = "DXY_U6"  # Dollar Index CFD on the user's IC Markets MT5

    # --- DXY concurrency filter (gold) ---
    # The hard rule at the heart of the strategy: the robot only signals
    # LONG gold when the dollar is weak (gauge high) and SHORT gold when
    # the dollar is strong (gauge low). Gauge: 100 = USD weak (bullish gold).
    dxy_filter_enabled: bool = True
    dxy_long_min: float = 55.0   # LONG needs gauge >= this
    dxy_short_max: float = 45.0  # SHORT needs gauge <= this

    # --- Notifications (Telegram) ---
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Trading sessions (analysis gate) ---
    # Analysis runs only while London or New York is open. Times are LOCAL
    # times; zoneinfo applies summer/winter offsets automatically.
    # Position management stays 24/7 — this gate only skips analysis.
    session_filter_enabled: bool = True
    session_london: str = "08:00-17:00"
    session_new_york: str = "09:30-17:00"

    # --- Multi-timeframe (pro logic) ---
    # TIMEFRAME (15m) is the ENTRY timeframe: triggers, stops, targets.
    # HTF_TIMEFRAME (4h) is the ANALYSIS timeframe: the directional bias.
    # LONG only in an HTF uptrend, SHORT only in an HTF downtrend; a choppy
    # HTF (ADX below htf_adx_min) blocks all trades.
    htf_timeframe: str = "4h"
    htf_bias_filter_enabled: bool = True
    htf_adx_min: float = 20.0

    # --- PAXG 24/7 fallback (weekends / stale futures) ---
    # Exchanges tried in order until one answers. Binance blocks US
    # datacenter IPs (Railway), so Kraken takes over automatically there.
    paxg_exchanges: list[str] = ["binance", "kraken", "okx"]

    # --- Risk (paper) ---
    paper_starting_equity: float = 10_000.0
    risk_per_trade: float = 0.01
    max_positions: int = 3
    max_exposure: float = 0.25
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.10
    atr_stop_mult: float = 2.0
    take_profit_rr: float = 2.0
    min_confidence: float = 0.55
    fee_rate: float = 0.001
    slippage: float = 0.0005

    # --- Signal fusion ---
    weight_technical: float = 0.45
    weight_regime: float = 0.35
    weight_sentiment: float = 0.20
    side_threshold: float = 0.25

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalise_symbols(cls, value: object) -> str:
        return ",".join(part.strip() for part in str(value).split(",") if part.strip())

    @property
    def symbol_list(self) -> list[str]:
        return [s for s in self.symbols.split(",") if s]

    @property
    def llm_enabled(self) -> bool:
        """True when a real-looking DeepSeek key is configured."""
        key = self.deepseek_api_key.get_secret_value() if self.deepseek_api_key else ""
        return bool(key) and not key.startswith("sk-your")

    @property
    def live_mode(self) -> bool:
        return self.execution_mode == "mt5"

    @property
    def mt5_configured(self) -> bool:
        secret = self.mt5_password.get_secret_value() if self.mt5_password else ""
        return bool(self.mt5_login) and bool(secret)


@lru_cache
def get_settings() -> Settings:
    return Settings()
