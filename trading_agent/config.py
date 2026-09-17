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
    # Push for REJECTED opportunities (§38): a refusal + its reason is
    # trader intelligence (why not to put money, and proof the robot is
    # still analysing). Consecutive refusals by the SAME gate are
    # deduplicated — one availability heartbeat per repeat window instead
    # of ~96 identical messages per day. Rejections always stay in the DB.
    telegram_rejection_alerts: bool = True
    telegram_rejection_repeat_minutes: int = 60

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
    # SYDNEY window (Sydney local) — a third analysis GATE alongside
    # London/NY: it opens the FX week (Sunday evening UTC) and covers the
    # post-New-York window. Set SESSION_SYDNEY="" to disable it.
    session_sydney: str = "07:00-16:00"

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
    # Phase A (V-MONSTER): market timestamps ahead of the local clock by
    # more than this = provider/clock skew -> degraded (clock integrity).
    data_clock_tolerance_s: int = 60
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
    # Deterministic 8-component score (mtf/structure/regime/dxy/
    # volatility/session/risk_reward/location). Proposals below this are
    # refused with no_trade_reason=LOW_SETUP_QUALITY. `location` is the
    # Phase C component (V-MONSTER §28): liquidity proximity, VWAP
    # relation, premium/discount, FVG/OB support.
    setup_quality_min: float = 0.45

    # --- Room-to-target gate (V-MONSTER §29, Phase C) ---
    # The distance to the opposing liquidity level, after spread and
    # slippage costs, must leave at least room_min_rr R of room —
    # otherwise INSUFFICIENT_ROOM refuses the proposal. Only fires when
    # the liquidity map found an opposing level; no data never blocks.
    room_gate_enabled: bool = True
    room_min_rr: float = 1.0

    # --- Market speed (V-MONSTER §27, Phase D) ---
    # Deterministic SLOW/NORMAL/FAST/EXTREME classification from
    # range-per-minute vs ATR, candle formation speed and volatility
    # acceleration. EXTREME refuses new entries only when opted in
    # (no_trade_extreme_speed) with ABNORMAL_SPEED.
    speed_window: int = 12  # baseline candles (current excluded)
    speed_fast_mult: float = 1.8  # range/formation ratio -> FAST
    speed_extreme_mult: float = 3.0  # ratio -> EXTREME
    speed_slow_mult: float = 0.4  # ratio <= this, not accelerating -> SLOW
    speed_accel_lookback: int = 12  # ATR acceleration window (candles)
    speed_accel_extreme_mult: float = 1.5  # accel on top of FAST -> EXTREME
    no_trade_extreme_speed: bool = False

    # --- Trigger quality (V-MONSTER §30, Phase D) ---
    # TRIGGER_QUALITY (0-1, BOS/sweep/displacement/zone shelter) is
    # computed in the fusion layer and stored with every signal; the
    # TRIGGER_SPEED state is kept separate. A trigger below
    # trigger_confirm_min, or one firing in EXTREME speed, is recorded
    # as not confirmed — enrichment, never a hard gate by itself.
    trigger_confirm_min: float = 0.6
    trigger_event_window: int = 12  # recency window for structure events

    # --- Opportunity clustering + dedup (V-MONSTER §31/§40/§41/§52/§53, Phase E) ---
    # One OPPORTUNITY_ID = direction + structure event + time proximity,
    # derived deterministically from the snapshot. The dedup gate
    # suppresses re-signals of the same opportunity: a pending stronger
    # signal blocks weaker re-signals (OPPORTUNITY_ACTIVE), and any
    # recent proposal at a close price is SIGNAL_DUPLICATE. No anchor,
    # no data or a DB failure never blocks (honesty, §4).
    opportunity_dedup_enabled: bool = True
    opportunity_dedup_window_minutes: int = 180  # time proximity
    opportunity_price_tolerance_pct: float = 0.1  # "close price" band
    opportunity_ttl_minutes: int = 720  # FORMING -> EXPIRED lifetime

    # --- Signal stability (V-MONSTER §32, Phase F) ---
    # STABLE/FRAGILE/VERY_FRAGILE from ±1 tick / −1 candle
    # perturbations. Sensitivity is measured in score deltas (tick) and
    # in ATRs (entry/stop); a speed-state flip on the shortened frame
    # marks a regime boundary. Enrichment, never a hard gate by itself.
    stability_tick: float = 0.01
    stability_score_fragile: float = 0.05  # setup-score delta -> FRAGILE
    stability_score_very_fragile: float = 0.15
    stability_entry_fragile_atr_mult: float = 0.2  # close move -> FRAGILE
    stability_entry_very_fragile_atr_mult: float = 0.4
    stability_stop_fragile_atr_mult: float = 0.1  # stop move -> FRAGILE
    stability_stop_very_fragile_atr_mult: float = 0.25

    # --- Final real-time revalidation (V-MONSTER §56, Phase F) ---
    # After all gates and before the send, one fresh tick must still
    # support the proposal: price drift, spread and data age are
    # re-checked. A failure aborts the send (recorded, no Telegram);
    # no tick source, no data or a fetch error never blocks.
    final_revalidation_enabled: bool = True
    revalidate_max_drift_pct: float = 0.15  # |tick - entry| / entry * 100
    revalidate_max_age_s: float = 600  # tick older than this = stale

    # --- Signal timing + actionability (V-MONSTER §42-§49/§64, Phase G) ---
    # TIMING_QUALITY 0-1 (trigger maturity, speed, remaining room,
    # drift, lifecycle); SIGNAL_LEAD_TIME from the speed-adjusted pace;
    # ACTIONABILITY_DEADLINE = now + lead; EXPECTED_EXECUTION_PRICE/
    # DRIFT over the human reaction window (reaction + Telegram latency
    # + spread). TOO_LATE refuses the send when the expected lead time
    # is shorter than the reaction window.
    timing_gate_enabled: bool = True
    user_reaction_seconds: float = 120.0  # human reaction budget
    telegram_latency_s: float = 3.0  # delivery latency on top
    max_chase_atr_mult: float = 0.5  # execution-zone half-width, 0 = disabled

    # --- Confidence tiers + A+/A/NO TRADE (V-MONSTER §58/§59/§81, Phase H) ---
    # HIGH needs a calibrated win rate (historical outcomes in the
    # signal's own confidence bucket) at/above the floor; LOW (below the
    # ceiling) refuses the proposal; uncalibrated history is MEDIUM and
    # is capped in size. A+ = top structural + liquidity + timing +
    # statistical bucket; absolute position/exposure limits unchanged.
    tier_high_min_calibrated: float = 0.6  # calibrated win rate -> HIGH
    tier_low_max_calibrated: float = 0.45  # below this -> LOW (reject)
    tier_medium_size_cap: float = 0.75  # MEDIUM size fraction (HIGH = 1.0)
    a_plus_setup_quality_min: float = 0.7  # structural floor for A+
    a_plus_timing_min: float = 0.7  # timing floor for A+
    a_plus_room_min_r: float = 2.0  # room-to-target floor for A+

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

    # --- Cost control (spec §48) ---
    # Pre-AI checks, all free: kill-switch, positive price/ATR and a
    # spread ceiling. When the ceiling is exceeded the cycle skips the
    # DeepSeek calls entirely (no API cost on clearly invalid markets).
    # 0 = spread ceiling disabled.
    ai_skip_max_spread_pct: float = 0.0

    # --- Economic calendar (INTELLIGENCE_V2 — phase 8, spec §43) ---
    # Optional by design: events only ever come from a real provider
    # (never from the LLM). "null" = offline (no events, no blocking).
    # When enabled, a HIGH-importance USD event within
    # news_block_minutes refuses new entries with NEWS_RISK.
    news_filter_enabled: bool = False
    news_provider: str = "null"  # "null" | "finnhub"
    finnhub_api_key: str = ""  # free read-only key from finnhub.io
    news_block_minutes: int = 30
    news_min_importance: str = "HIGH"  # HIGH | MEDIUM | LOW

    # --- Market shock detection (INTELLIGENCE_V2 — phase 8, spec §44) ---
    # Deterministic classification from candle range / ATR / volume /
    # price movement / spread versus a rolling baseline: NORMAL,
    # VOLATILITY_EXPANSION (warning) or SHOCK (blocks new entries).
    # A SHOCK also starts a persisted cooldown (shock_cooldown_minutes).
    shock_enabled: bool = True
    shock_lookback: int = 60  # baseline candles (current excluded)
    shock_expansion_multiple: float = 1.8  # axis ratio -> VOLATILITY_EXPANSION
    shock_multiple: float = 3.0  # axis ratio -> SHOCK
    shock_movement_pct: float = 1.0  # body/gap % of price -> SHOCK
    shock_spread_pct: float = 0.0  # spread % of price -> SHOCK (0 = off)
    shock_block_new_entries: bool = True
    shock_cooldown_minutes: int = 30  # keep blocking after a SHOCK (0 = off)

    # --- Statistical quality gate (INTELLIGENCE_V2 — phase 6, spec §31/§32) ---
    # Compares the candidate against resolved signals sharing its
    # side/regime/DXY class. Below min_sample_for_statistics the quality
    # is UNKNOWN and never blocks (spec §31). With a sufficient sample
    # whose historical expectancy is below the configured floor the
    # signal is refused with STATISTICAL_QUALITY (spec §32). Opt-in.
    min_sample_for_statistics: int = 30
    statistical_quality_enabled: bool = False
    statistical_quality_min_expectancy: float = 0.0

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
