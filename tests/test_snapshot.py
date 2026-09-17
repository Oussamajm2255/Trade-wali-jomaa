"""Canonical snapshot builder (spec §3): one coherent snapshot per cycle."""
from __future__ import annotations

import pandas as pd
import pytest

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.data.market import MarketDataError
from trading_agent.data.quality import QualityState
from trading_agent.data.snapshot import build_market_snapshot
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection


def make_df(n: int = 300, timeframe: str = "15m", bad: bool = False) -> pd.DataFrame:
    freq = {"15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}[timeframe]
    end = pd.Timestamp.now(tz="UTC").floor(freq)
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    close = 2400.0 + pd.Series(range(n), dtype=float).to_numpy() * 0.01
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 100.0},
        index=idx,
    )
    if bad:
        df.loc[df.index[10], "high"] = df.loc[df.index[10], "low"] - 1.0
    return df


def fresh_gauge() -> dict:
    return {
        "value": 60,
        "classification": "Bullish (USD weak)",
        "kind": "dxy",
        "ts": str(pd.Timestamp.now(tz="UTC")),
    }


class FakeMarket:
    def __init__(
        self,
        gauge: dict | None = None,
        fail_tfs: tuple[str, ...] = (),
        bad_entry: bool = False,
        source: str = "yfinance:GC=F",
        dxy_df: pd.DataFrame | None = None,
    ) -> None:
        self.gauge = gauge
        self.fail_tfs = set(fail_tfs)
        self.bad_entry = bad_entry
        self.last_source = source
        self.dxy_df = dxy_df

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if timeframe in self.fail_tfs:
            raise RuntimeError(f"fake {timeframe} down")
        return make_df(300, timeframe, bad=self.bad_entry and timeframe == "15m")

    def sentiment_gauge(self) -> dict | None:
        return self.gauge

    def dxy_ohlcv(self, timeframe: str, limit: int) -> pd.DataFrame:
        if self.dxy_df is None:
            raise RuntimeError("fake DXY candles down")
        return self.dxy_df


def make_settings(**overrides) -> Settings:
    return Settings(
        htf_timeframe="4h",
        snapshot_timeframes=["1h", "4h", "1d"],
        telegram_bot_token="",
        telegram_chat_id="",
        **overrides,
    )


def test_builds_all_timeframes_with_pass_quality() -> None:
    gauge = fresh_gauge()
    snap = build_market_snapshot(FakeMarket(gauge=gauge), "XAUUSD", make_settings(), "15m")
    assert set(snap.candles) == {"15m", "1h", "4h", "1d"}
    assert set(snap.biases) == {"15m", "1h", "4h", "1d"}
    assert snap.quality_state == QualityState.PASS
    assert snap.price == snap.indicators["15m"]["last_close"]
    assert snap.indicators["15m"]["rsi_14"] is not None
    assert snap.indicators["4h"]["ema50_gt_ema200"] is True  # rising close
    assert snap.session["in_session"] in (True, False)
    assert snap.versions["strategy_version"] == "LEGACY_BASELINE"
    assert snap.dxy == gauge


def test_snapshot_carries_phase2_deterministic_context() -> None:
    snap = build_market_snapshot(FakeMarket(gauge=fresh_gauge()), "XAUUSD", make_settings(), "15m")
    assert set(snap.regimes) == {"15m", "1h", "4h", "1d"}
    assert snap.regimes["15m"]["regime"] in (
        "trend_up", "trend_down", "range", "high_volatility", "low_volatility", "transition"
    )
    assert snap.alignment["alignment"] in ("BULLISH_ALIGNMENT", "BEARISH_ALIGNMENT", "MIXED", "CONFLICTED")
    assert snap.structure["timeframe"] == "15m"
    assert snap.structure["support"] is not None
    # No intraday DXY candles on the fake -> honest gauge-only context.
    assert snap.dxy_context["kind"] == "dxy_context"
    assert snap.dxy_context["source"] == "gauge only"
    assert snap.gold_context["daily_open"] is not None
    assert snap.gold_context["prev_day_high"] is not None
    assert snap.session_context["session"] in (
        "ASIA", "SYDNEY", "LONDON", "LONDON_NY_OVERLAP", "NEW_YORK", "OFF_SESSION"
    )
    entry = snap.entry_snapshot_for_llm("4h")
    assert "mtf_biases" in entry and "alignment" in entry
    assert entry["structure"]["timeframe"] == "15m"
    assert entry["regime"]["regime"] == snap.regimes["15m"]["regime"]
    assert "1h" in entry["htf_regimes"]
    assert entry["gold_context"]["daily_open"] is not None
    assert entry["dxy_context"]["source"] == "gauge only"


def test_snapshot_carries_phase_a_latency_metrics() -> None:
    snap = build_market_snapshot(FakeMarket(gauge=fresh_gauge()), "XAUUSD", make_settings(), "15m")
    assert snap.data_latency_ms >= 0
    assert snap.data_age_s is not None and snap.data_age_s >= 0
    entry = snap.entry_snapshot_for_llm()
    # Metrics ride into the LLM dict -> stored with every signal record.
    assert entry["data_age_s"] == snap.data_age_s
    assert entry["data_latency_ms"] == round(snap.data_latency_ms, 1)


def test_snapshot_carries_phase_b_liquidity_and_vwap() -> None:
    snap = build_market_snapshot(FakeMarket(gauge=fresh_gauge()), "XAUUSD", make_settings(), "15m")
    assert snap.liquidity["price"] == snap.price
    assert snap.liquidity["levels"]  # PDH/PDL always present via the 1d frame
    assert 0.0 <= snap.liquidity["quality"] <= 1.0
    assert snap.vwap["available"] is True  # yfinance source -> volume trusted
    assert snap.vwap["daily_vwap"] is not None
    assert snap.vwap["state"] in ("above", "below", "reclaimed", "rejected")
    entry = snap.entry_snapshot_for_llm()
    assert entry["liquidity"] is snap.liquidity
    assert entry["vwap"] is snap.vwap
    ctx = snap.context_for_risk()
    assert ctx["liquidity"] is snap.liquidity
    assert ctx["vwap"] is snap.vwap


def test_proxy_source_marks_vwap_unavailable() -> None:
    snap = build_market_snapshot(
        FakeMarket(gauge=fresh_gauge(), source="PAXG/USDT proxy (futures closed)"),
        "XAUUSD",
        make_settings(),
        "15m",
    )
    assert snap.vwap["available"] is False
    assert "proxy" in snap.vwap["reason"]
    assert snap.vwap["daily_vwap"] is None
    # Liquidity levels stay honest on proxy data too.
    assert snap.liquidity["price"] == snap.price


def test_snapshot_with_intraday_dxy_candles() -> None:
    dxy = make_df(100, "15m")
    snap = build_market_snapshot(
        FakeMarket(gauge=fresh_gauge(), dxy_df=dxy), "XAUUSD", make_settings(), "15m"
    )
    assert snap.dxy_context["source"] == "intraday candles"
    assert snap.dxy_context["level"] is not None
    assert snap.dxy_context["change_15m_pct"] is not None
    # Phase 3: XAUUSD response/divergence computed deterministically (spec §14).
    assert snap.dxy_context["xau_vs_dxy"] is not None
    assert "relationship_1h" in snap.dxy_context["xau_vs_dxy"]


def test_entry_fetch_failure_raises() -> None:
    with pytest.raises(MarketDataError):
        build_market_snapshot(FakeMarket(fail_tfs=("15m",)), "XAUUSD", make_settings(), "15m")


def test_entry_invalid_frame_raises() -> None:
    with pytest.raises(MarketDataError) as exc_info:
        build_market_snapshot(FakeMarket(bad_entry=True), "XAUUSD", make_settings(), "15m")
    assert "invalid" in str(exc_info.value)


def test_htf_failure_fails_closed_when_bias_gate_enabled() -> None:
    snap = build_market_snapshot(
        FakeMarket(gauge=fresh_gauge(), fail_tfs=("4h",)), "XAUUSD", make_settings(), "15m"
    )
    assert snap.quality_state == QualityState.FAIL
    assert any("fetch 4h" in i for i in snap.quality_issues)
    assert "4h" not in snap.biases  # fail closed: no bias to pass the gate


def test_htf_failure_only_degrades_when_gate_disabled() -> None:
    snap = build_market_snapshot(
        FakeMarket(gauge=fresh_gauge(), fail_tfs=("4h",)),
        "XAUUSD",
        make_settings(htf_bias_filter_enabled=False),
        "15m",
    )
    assert snap.quality_state == QualityState.DEGRADED
    assert any("fetch 4h" in i for i in snap.quality_issues)


def test_missing_gauge_labels_degraded() -> None:
    snap = build_market_snapshot(FakeMarket(gauge=None), "XAUUSD", make_settings(), "15m")
    assert snap.quality_state == QualityState.DEGRADED
    assert any("dxy_unavailable" in i for i in snap.quality_issues)
    entry = snap.entry_snapshot_for_llm("4h")
    assert entry["data_quality"] == "degraded"
    assert "data_quality_issues" in entry


def test_provider_fallback_labels_degraded() -> None:
    snap = build_market_snapshot(
        FakeMarket(gauge=fresh_gauge(), source="PAXG/USDT proxy (futures closed)"),
        "XAUUSD",
        make_settings(),
        "15m",
    )
    assert snap.quality_state == QualityState.DEGRADED
    assert any("proxy" in i for i in snap.quality_issues)


def test_entry_llm_snapshot_carries_htf_bias_subset() -> None:
    snap = build_market_snapshot(FakeMarket(gauge=fresh_gauge()), "XAUUSD", make_settings(), "15m")
    entry = snap.entry_snapshot_for_llm("4h")
    assert entry["htf_bias"]["timeframe"] == "4h"
    assert entry["htf_bias"]["bias"] in ("bull", "bear", "neutral")
    assert "ema50" not in entry["htf_bias"]  # subset only, like v1


def test_context_for_risk_has_quality_session_versions() -> None:
    snap = build_market_snapshot(FakeMarket(gauge=fresh_gauge()), "XAUUSD", make_settings(), "15m")
    ctx = snap.context_for_risk()
    assert ctx["data_quality"]["state"] == "pass"
    assert "in_session" in ctx["session"]
    assert ctx["versions"]["risk_engine_version"]


def test_orchestrator_fails_closed_without_running_agents() -> None:
    """Data-quality FAIL must reject before any agent (and any AI call) runs."""
    settings = make_settings()
    market = FakeMarket(gauge=fresh_gauge(), fail_tfs=("4h",))
    orch = Orchestrator(settings, market, RiskEngine(settings))
    result, verdicts, snapshot, gauge = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    assert "data quality FAIL" in result.reason
    assert verdicts == {}  # no agent ever ran -> no AI calls on invalid data


def test_orchestrator_degrades_but_continues_when_allowed() -> None:
    settings = make_settings()
    market = FakeMarket(gauge=None)  # degraded: DXY missing
    orch = Orchestrator(settings, market, RiskEngine(settings))
    result, verdicts, snapshot, gauge = orch.run_full("XAUUSD", "15m")
    # Degraded is allowed by default -> agents run (fallback, no LLM key).
    assert verdicts != {}
    assert snapshot["data_quality"] == "degraded"
