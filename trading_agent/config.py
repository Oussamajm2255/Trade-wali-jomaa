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
    # ASIA window (Tokyo local) — used only for the session CLASSIFICATION
    # stored with every signal (spec §11); it never gates analysis.
    session_asia: str = "09:00-18:00"

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

    # --- DeepSeek balance monitoring ---
    # Checked once per UTC day; a Telegram alert is sent when the USD
    # balance falls below this threshold (0 balance = degraded mode).
    deepseek_balance_warn_usd: float = 1.0

    # --- Canonical snapshot (INTELLIGENCE_V2 — phase 1) ---
    # One coherent snapshot per analysis cycle: the entry timeframe plus
    # these higher timeframes, each validated, with indicators + bias.
    # All downstream modules consume this snapshot instead of fetching
    # or computing overlapping data independently.
    snapshot_timeframes: list[str] = ["1h", "4h", "1d"]

    # --- Data quality validation ---
    # FAIL stops the cycle (no AI calls, no proposal). DEGRADED continues
    # only when data_quality_allow_degraded is true, and is clearly
    # labelled and stored on every signal and rejected opportunity.
    data_quality_min_candles: int = 60
    data_quality_max_stale_multiple: float = 3.0  # last candle age vs TF duration
    data_quality_allow_degraded: bool = True
    dxy_max_age_hours: int = 48  # older DXY gauge = degraded

    # --- Deterministic regime engine (INTELLIGENCE_V2 — phase 2) ---
    # TREND_UP / TREND_DOWN / RANGE / HIGH_VOLATILITY / LOW_VOLATILITY /
    # TRANSITION, classified from ADX, ATR behaviour and EMA alignment.
    regime_adx_trend: float = 25.0
    regime_atr_high_mult: float = 1.8  # ATR >= 1.8x its median = high vol
    regime_atr_low_mult: float = 0.55  # ATR <= 0.55x its median = low vol

    # --- Market structure / SMC detection ---
    structure_swing_left: int = 3
    structure_swing_right: int = 3
    structure_tolerance_pct: float = 0.05  # equal highs/lows clustering
    structure_fvg_min_atr_mult: float = 0.3  # minimum FVG gap vs ATR
    structure_displacement_atr_mult: float = 1.5  # displacement candle range
    structure_sweep_lookback: int = 30

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
    weight_sentiment: float = 0.20  # used by the DXY context agent on gold
    side_threshold: float = 0.25

    # --- AI failure isolation (INTELLIGENCE_V2 — phase 3, spec §37) ---
    # A failed agent = LLM call failed (TIMEOUT / INVALID_JSON / API_ERROR /
    # RATE_LIMIT / EMPTY_RESPONSE) and its heuristic fallback was used.
    # 0 failed = normal, 1 = degraded warning, N >= block_min = no new
    # proposals. LLM-disabled-by-config is NOT counted as a failure.
    agent_failure_block_enabled: bool = True
    agent_failure_block_min: int = 2

    # --- AI reliability tracking (spec §15) ---
    # Rolling window for per-agent statistics; analysis-only, weights are
    # never modified from this data.
    agent_reliability_window: int = 200

    # --- Setup quality engine (INTELLIGENCE_V2 — phase 4, spec §18) ---
    # Deterministic 7-component score (mtf/structure/regime/dxy/
    # volatility/session/risk_reward). Proposals below this are refused
    # with no_trade_reason=LOW_SETUP_QUALITY.
    setup_quality_min: float = 0.45

    # --- Conflict detection (spec §19) ---
    # Contradictions between the fused direction and the deterministic
    # context axes. N >= conflict_conflicted_min conflicts -> CONFLICTED.
    # CONFLICTED blocks new proposals by default; when blocking is off it
    # takes the capped quality penalty instead. MIXED always reduces the
    # setup-quality score by conflict_penalty_per_axis per conflict.
    conflict_conflicted_min: int = 2
    conflict_block_conflicted: bool = True
    conflict_penalty_per_axis: float = 0.15
    conflict_max_penalty: float = 0.4

    # --- No-trade gates (spec §20) ---
    # HIGH_VOLATILITY is opt-in (phase 2 shipped high-vol without a
    # block). BAD_SPREAD only fires when the data provider supplies a
    # spread and the threshold is > 0 (MT5 gauge). STATISTICAL_EDGE_
    # UNKNOWN must stay off until the outcome engine (phase 5) produces
    # calibrated confidence.
    no_trade_high_volatility: bool = False
    no_trade_max_spread_pct: float = 0.0
    require_statistical_edge: bool = False

    # --- Confidence calibration (spec §21) ---
    # calibrated_confidence = historical win rate inside the signal's own
    # 0.05-wide raw-confidence bucket, and only once BOTH sample minimums
    # hold. Until then it is None and the product must not claim it.
    calibration_window: int = 500
    calibration_min_samples: int = 50
    calibration_min_per_bin: int = 10

    # --- Historical backtesting (INTELLIGENCE_V2 — phase 5, spec §24/§25) ---
    # Candle-by-candle replay of the full pipeline on pre-fetched history.
    # Deterministic AI mode: the backtest always runs without an API key
    # (heuristic fallbacks), so results are reproducible and offline.
    backtest_history_limit: int = 5000  # candles fetched per timeframe
    backtest_db_url: str = "sqlite:///backtest.db"  # isolated risk/DB state
    backtest_spread_pct: float = 0.0  # round-trip spread (half paid each side)

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
