"""Canonical market snapshot — ONE coherent snapshot per analysis cycle.

Spec §3: every downstream module (agents, fusion, risk engine) consumes
this object instead of independently fetching or calculating overlapping
information. The AI never touches market access: it only reads the
pre-computed numbers assembled here. Deterministic fields (biases,
session, data quality) are computed locally, never by the LLM.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

from trading_agent.config import Settings
from trading_agent.data.alignment import classify_alignment
from trading_agent.data.bias import compute_htf_bias
from trading_agent.data.calendar import build_calendar_provider
from trading_agent.data.dxy_context import compute_dxy_context, xau_vs_dxy
from trading_agent.data.gold_context import compute_gold_context
from trading_agent.data.indicators import adx, atr, build_snapshot, ema, rsi
from trading_agent.data.liquidity import compute_liquidity
from trading_agent.data.market import MarketDataError
from trading_agent.data.quality import (
    DataQuality,
    QualityState,
    combine_quality,
    validate_candles,
    validate_gauge,
    validate_indicators,
)
from trading_agent.data.regime import detect_regime
from trading_agent.data.sessions import (
    session_context,
    session_start_utc,
    session_state,
)
from trading_agent.data.shock import detect_shock
from trading_agent.data.speed import compute_market_speed
from trading_agent.data.structure import detect_structure
from trading_agent.data.vwap import compute_vwap
from trading_agent.versioning import version_stamp

logger = logging.getLogger(__name__)


class MarketSnapshot(BaseModel):
    """All market knowledge for one analysis cycle, computed once."""

    model_config = {"arbitrary_types_allowed": True}

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    entry_timeframe: str
    price: float = 0.0
    data_source: str = "unknown"
    candles: dict[str, pd.DataFrame] = Field(default_factory=dict)
    indicators: dict[str, dict] = Field(default_factory=dict)
    biases: dict[str, dict] = Field(default_factory=dict)
    regimes: dict[str, dict] = Field(default_factory=dict)
    alignment: dict = Field(default_factory=dict)
    structure: dict = Field(default_factory=dict)
    dxy: dict | None = None
    dxy_context: dict | None = None
    gold_context: dict = Field(default_factory=dict)
    session: dict = Field(default_factory=dict)
    session_context: dict = Field(default_factory=dict)
    # Phase 8: normalized economic-calendar events ahead of this cycle
    # (spec §43, empty = provider offline) and the deterministic shock
    # classification (spec §44).
    news_context: list[dict] = Field(default_factory=list)
    shock_context: dict = Field(default_factory=dict)
    # Phase B (V-MONSTER §9/§12): deterministic liquidity map + VWAP
    # anchors, both built from the snapshot's own candles.
    liquidity: dict = Field(default_factory=dict)
    vwap: dict = Field(default_factory=dict)
    # Phase D (V-MONSTER §27): deterministic market speed
    # (SLOW/NORMAL/FAST/EXTREME); the side-aware trigger quality (§30)
    # lives in the fusion context instead.
    speed: dict = Field(default_factory=dict)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    versions: dict = Field(default_factory=dict)
    # Phase A (V-MONSTER §5): latency/age metrics for one cycle — how
    # long the data took to acquire and how fresh the newest candle is.
    data_latency_ms: float = 0.0
    data_age_s: float | None = None

    @property
    def quality_state(self) -> QualityState:
        return self.data_quality.state

    @property
    def quality_issues(self) -> list[str]:
        return self.data_quality.issues

    @property
    def degraded(self) -> bool:
        return self.data_quality.state == QualityState.DEGRADED

    def entry_snapshot_for_llm(self, htf_timeframe: str | None = None) -> dict:
        """JSON-safe dict handed to the agents (v1 shape + canonical context)."""
        snap = dict(self.indicators.get(self.entry_timeframe, {}))
        snap["symbol"] = self.symbol
        snap["timeframe"] = self.entry_timeframe
        snap["data_source"] = self.data_source
        snap["session"] = self.session
        snap["session_context"] = self.session_context
        snap["data_quality"] = self.data_quality.state.value
        if self.data_quality.issues:
            snap["data_quality_issues"] = self.data_quality.issues
        # Phase A metrics ride along: stored with every signal record
        # (spec §85 lists system latency as required persistence).
        snap["data_age_s"] = self.data_age_s
        snap["data_latency_ms"] = round(self.data_latency_ms, 1)
        if htf_timeframe and htf_timeframe in self.biases:
            bias = self.biases[htf_timeframe]
            snap["htf_bias"] = {
                "timeframe": htf_timeframe,
                "bias": bias["bias"],
                "detail": bias["detail"],
                "adx": bias.get("adx"),
            }
        snap["mtf_biases"] = {
            tf: {"bias": b["bias"], "detail": b["detail"]} for tf, b in self.biases.items()
        }
        snap["alignment"] = self.alignment
        snap["regime"] = self.regimes.get(self.entry_timeframe, {})
        snap["htf_regimes"] = {
            tf: r for tf, r in self.regimes.items() if tf != self.entry_timeframe
        }
        snap["structure"] = self.structure
        snap["gold_context"] = self.gold_context
        snap["dxy_context"] = self.dxy_context
        snap["news_context"] = self.news_context
        snap["shock_context"] = self.shock_context
        snap["liquidity"] = self.liquidity
        snap["vwap"] = self.vwap
        snap["speed"] = self.speed
        return snap

    def context_for_risk(self) -> dict:
        """Deterministic context stored with every proposal (never AI-built)."""
        return {
            "data_quality": {
                "state": self.data_quality.state.value,
                "issues": self.data_quality.issues,
            },
            "session": self.session,
            "session_context": self.session_context,
            "alignment": self.alignment,
            "regime": self.regimes.get(self.entry_timeframe, {}),
            "dxy_context": self.dxy_context,
            "gold_context": self.gold_context,
            "news_context": self.news_context,
            "shock_context": self.shock_context,
            "liquidity": self.liquidity,
            "vwap": self.vwap,
            "speed": self.speed,
            "versions": self.versions,
        }


def gauge_callable(market: Any) -> Callable[[], dict | None]:
    """Pick the sentiment/DXY source of a market provider (gold vs crypto)."""
    if hasattr(market, "sentiment_gauge"):
        return market.sentiment_gauge
    if hasattr(market, "fear_greed_index"):
        return market.fear_greed_index
    return lambda: None


def compact_indicators(df: pd.DataFrame) -> dict:
    """Smaller indicator set for higher timeframes (context, not entries)."""
    close = df["close"]
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    ema200 = ema(close, 200)
    return {
        "last_close": round(float(close.iloc[-1]), 8),
        "rsi_14": round(float(rsi(close).iloc[-1]), 2),
        "adx_14": round(float(adx(df).iloc[-1]), 2),
        "atr_14": round(float(atr(df).iloc[-1]), 8),
        "ema20": round(float(ema20.iloc[-1]), 8),
        "ema50": round(float(ema50.iloc[-1]), 8),
        "ema200": round(float(ema200.iloc[-1]), 8),
        "ema20_gt_ema50": bool(ema20.iloc[-1] > ema50.iloc[-1]),
        "ema50_gt_ema200": bool(ema50.iloc[-1] > ema200.iloc[-1]),
    }


def build_market_snapshot(
    market: Any,
    symbol: str,
    settings: Settings,
    entry_timeframe: str | None = None,
    now: datetime | None = None,
) -> MarketSnapshot:
    """Fetch, validate and compute everything once; the cycle's source of truth.

    Raises MarketDataError when the entry timeframe cannot be analysed —
    that is a FAIL: no AI calls, no proposal (§4 / §48 cost control).

    `now` defaults to wall-clock time; historical replays (backtesting)
    pass the replay timestamp so every time-dependent check (staleness,
    session classification) reflects the moment the signal was made.
    """
    entry_tf = entry_timeframe or settings.timeframe
    timeframes = [entry_tf] + [tf for tf in settings.snapshot_timeframes if tf != entry_tf]

    snap = MarketSnapshot(
        symbol=symbol,
        entry_timeframe=entry_tf,
        timestamp=now or datetime.now(timezone.utc),
        versions=version_stamp(),
    )
    qualities: list[DataQuality] = []
    entry_age: float | None = None

    t_fetch = time.monotonic()
    for tf in timeframes:
        try:
            df = market.fetch_ohlcv(symbol, tf, settings.ohlcv_limit)
        except Exception as exc:  # noqa: BLE001 - provider errors surface as quality
            if tf == entry_tf:
                raise MarketDataError(f"entry timeframe {tf} fetch failed: {exc}") from exc
            # A missing HTF that backs the bias gate must fail closed;
            # other missing context only degrades.
            state = (
                QualityState.FAIL
                if settings.htf_bias_filter_enabled and tf == settings.htf_timeframe
                else QualityState.DEGRADED
            )
            qualities.append(DataQuality(state=state, issues=[f"fetch {tf} failed: {exc}"]))
            continue

        q = validate_candles(
            df,
            tf,
            min_candles=settings.data_quality_min_candles,
            max_stale_multiple=settings.data_quality_max_stale_multiple,
            now=now,
            clock_tolerance_s=settings.data_clock_tolerance_s,
        )
        qualities.append(q)
        if tf == entry_tf:
            entry_age = q.age_s
        if q.state == QualityState.FAIL:
            if tf == entry_tf:
                raise MarketDataError(f"entry timeframe {tf} invalid: {q.issues}")
            continue  # bad higher-TF frame: no indicators from garbage
        df = df[~df.index.duplicated(keep="last")]
        snap.candles[tf] = df
        if tf == entry_tf:
            snap.indicators[tf] = build_snapshot(df)
            snap.indicators[tf]["symbol"] = symbol
            snap.indicators[tf]["timeframe"] = tf
        else:
            snap.indicators[tf] = compact_indicators(df)
        snap.biases[tf] = compute_htf_bias(df, settings.htf_adx_min)
        try:
            snap.regimes[tf] = detect_regime(
                df,
                adx_trend=settings.regime_adx_trend,
                atr_high_mult=settings.regime_atr_high_mult,
                atr_low_mult=settings.regime_atr_low_mult,
            )
        except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
            logger.warning("regime detection failed on %s: %s", tf, exc)
            qualities.append(
                DataQuality(state=QualityState.DEGRADED, issues=[f"regime detection failed on {tf}"])
            )

    if entry_tf not in snap.indicators:
        raise MarketDataError("no valid entry timeframe snapshot")
    snap.data_latency_ms = (time.monotonic() - t_fetch) * 1000
    snap.data_age_s = entry_age
    snap.price = float(snap.indicators[entry_tf]["last_close"])
    snap.data_source = (
        getattr(market, "last_source", "") or getattr(market, "exchange_id", "unknown")
    )
    if "proxy" in snap.data_source.lower():
        qualities.append(
            DataQuality(state=QualityState.DEGRADED, issues=["provider fallback in use (proxy data)"])
        )
    # Volume only counts on the primary feed: on proxy data (PAXG token)
    # volume measures token flow, not gold flow — the shock engine and
    # the VWAP (Phase B) both respect this flag.
    trust_volume = "proxy" not in snap.data_source.lower()

    # Deterministic analysis layer (spec §5/§7): MTF alignment, entry-TF
    # structure. Enrichment — a failure degrades, never kills the cycle.
    try:
        snap.alignment = classify_alignment(snap.biases)
        snap.structure = detect_structure(
            snap.candles[entry_tf],
            left=settings.structure_swing_left,
            right=settings.structure_swing_right,
            tolerance_pct=settings.structure_tolerance_pct,
            fvg_min_atr_mult=settings.structure_fvg_min_atr_mult,
            displacement_atr_mult=settings.structure_displacement_atr_mult,
            sweep_lookback=settings.structure_sweep_lookback,
            timeframe=entry_tf,
        )
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
        logger.warning("deterministic analysis failed: %s", exc)
        qualities.append(
            DataQuality(state=QualityState.DEGRADED, issues=[f"deterministic analysis failed: {exc}"])
        )

    try:
        snap.dxy = gauge_callable(market)()
    except Exception as exc:  # noqa: BLE001 - gauge failure degrades, never kills
        logger.warning("sentiment gauge fetch failed: %s", exc)
        snap.dxy = None
    qualities.append(validate_gauge(snap.dxy, settings.dxy_max_age_hours, now=now))

    # Richer DXY context (§9) around the compatibility gauge: level,
    # direction, momentum, trend, volatility from real DXY candles.
    dxy_df = None
    if hasattr(market, "dxy_ohlcv"):
        try:
            dxy_df = market.dxy_ohlcv("15m", settings.ohlcv_limit)
        except Exception as exc:  # noqa: BLE001 - falls back to gauge-only
            logger.warning("intraday DXY candles unavailable: %s", exc)
    snap.dxy_context = compute_dxy_context(dxy_df, snap.dxy)
    if dxy_df is not None and snap.dxy_context is not None:
        try:
            snap.dxy_context["xau_vs_dxy"] = xau_vs_dxy(snap.candles[entry_tf], dxy_df)
        except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
            logger.warning("xau_vs_dxy failed: %s", exc)

    # Phase 8: shock classification (spec §44) — pure candle math, uses
    # the gauge spread when the provider supplies one. The economic
    # calendar (spec §43) is optional and always fails open: a broken
    # feed degrades to "no events", never to fabricated ones.
    try:
        gauge_spread = snap.dxy.get("spread") if isinstance(snap.dxy, dict) else None
        snap.shock_context = detect_shock(
            snap.candles[entry_tf],
            lookback=settings.shock_lookback,
            shock_multiple=settings.shock_multiple,
            expansion_multiple=settings.shock_expansion_multiple,
            movement_pct=settings.shock_movement_pct,
            spread_pct_threshold=settings.shock_spread_pct,
            spread=gauge_spread,
            trust_volume=trust_volume,
        )
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
        logger.warning("shock detection failed: %s", exc)
    try:
        provider = getattr(market, "calendar_provider", None) or build_calendar_provider(settings)
        snap.news_context = provider.upcoming_events(
            snap.timestamp, max(settings.news_block_minutes, 60)
        )
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
        logger.warning("economic calendar failed: %s", exc)

    qualities.append(validate_indicators(snap.indicators[entry_tf]))

    # Gold-specific key levels (§10) + session classification (§11).
    snap.session = session_state(
        now=now,
        london=settings.session_london,
        new_york=settings.session_new_york,
        sydney=settings.session_sydney,
    )
    snap.session_context = session_context(
        now=now,
        london=settings.session_london,
        new_york=settings.session_new_york,
        asia=settings.session_asia,
        sydney=settings.session_sydney,
    )
    session_start = session_start_utc(
        now=now,
        london=settings.session_london,
        new_york=settings.session_new_york,
        asia=settings.session_asia,
        sydney=settings.session_sydney,
    )
    snap.gold_context = compute_gold_context(
        snap.candles[entry_tf],
        daily_df=snap.candles.get("1d"),
        session_start=session_start,
    )

    # Phase B (V-MONSTER §9/§12): liquidity map + VWAP, built from the
    # snapshot's own candles — deterministic enrichment, never fatal.
    try:
        snap.liquidity = compute_liquidity(
            snap.candles[entry_tf],
            snap.gold_context,
            snap.structure,
            now=now,
            daily_df=snap.candles.get("1d"),
        )
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
        logger.warning("liquidity map failed: %s", exc)
    try:
        snap.vwap = compute_vwap(
            snap.candles[entry_tf],
            session_start=session_start,
            trust_volume=trust_volume,
        )
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
        logger.warning("vwap failed: %s", exc)

    # Phase D (V-MONSTER §27): market speed from the entry frame,
    # anchored to the cycle time (wall clock live, replay time in
    # backtests) so the formation ratio reflects the real candle.
    try:
        snap.speed = compute_market_speed(
            snap.candles[entry_tf],
            timeframe=entry_tf,
            window=settings.speed_window,
            fast_mult=settings.speed_fast_mult,
            extreme_mult=settings.speed_extreme_mult,
            slow_mult=settings.speed_slow_mult,
            accel_lookback=settings.speed_accel_lookback,
            accel_extreme_mult=settings.speed_accel_extreme_mult,
            now=now or snap.timestamp,
        )
    except Exception as exc:  # noqa: BLE001 - enrichment, never fatal
        logger.warning("market speed failed: %s", exc)

    snap.data_quality = combine_quality(*qualities)
    logger.info(
        "snapshot %s %s: quality=%s issues=%s",
        symbol,
        entry_tf,
        snap.data_quality.state.value,
        snap.data_quality.issues,
    )
    return snap
