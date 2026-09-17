"""Confidence tiers + A+/A/NO TRADE labels (V-MONSTER §58/§59/§81).

The guarantees under test:
- Tier assignment from the calibrated win rate: HIGH at/above
  `tier_high_min_calibrated`, LOW below `tier_low_max_calibrated`,
  MEDIUM otherwise — uncalibrated data is MEDIUM, never HIGH.
- The risk engine refuses LOW with a classified LOW_TIER rejection;
  MEDIUM is capped at `tier_medium_size_cap` while the absolute
  exposure/positions limits stay unchanged; HIGH trades full size.
- A+ requires the top of EVERY axis (statistical tier, structural
  quality, timing, liquidity room); everything else approved is A;
  rejections are NO TRADE — deterministic from the same inputs.
- The orchestrator stamps tier + label on the opportunity context and
  the signal record; Telegram renders the label on proposals and
  rejections.
"""
from __future__ import annotations

import pytest
from sqlalchemy import select

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.fusion.tier import (
    NO_TRADE_LABEL,
    ConfidenceTier,
    assign_tier,
    signal_label,
    tier_size_mult,
)
from trading_agent.fusion.types import (
    ConflictReport,
    ConflictState,
    FusionContext,
    FusionResult,
    NoTradeReason,
    SetupQuality,
)
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import AgentVerdict, Rejection, Side, SignalProposal
from trading_agent.store.db import session_scope
from trading_agent.store.models import SignalRecord

from test_snapshot import FakeMarket, fresh_gauge


# ------------------------------------------------------------- tier rules


def test_assign_tier_high_boundary() -> None:
    settings = Settings()
    assert assign_tier(0.6, settings) == ConfidenceTier.HIGH
    assert assign_tier(0.65, settings) == ConfidenceTier.HIGH
    assert assign_tier(1.0, settings) == ConfidenceTier.HIGH


def test_assign_tier_low_below_floor() -> None:
    settings = Settings()
    assert assign_tier(0.2, settings) == ConfidenceTier.LOW
    assert assign_tier(0.449, settings) == ConfidenceTier.LOW


def test_assign_tier_medium_band_and_uncalibrated() -> None:
    settings = Settings()
    assert assign_tier(None, settings) == ConfidenceTier.MEDIUM
    assert assign_tier(0.45, settings) == ConfidenceTier.MEDIUM
    assert assign_tier(0.5, settings) == ConfidenceTier.MEDIUM
    assert assign_tier(0.599, settings) == ConfidenceTier.MEDIUM


def test_tier_size_mult_defaults() -> None:
    settings = Settings()
    assert tier_size_mult(ConfidenceTier.HIGH, settings) == 1.0
    assert tier_size_mult(ConfidenceTier.MEDIUM, settings) == 0.75
    assert tier_size_mult(ConfidenceTier.LOW, settings) == 0.0


def test_tier_size_mult_cap_is_clamped_to_one() -> None:
    settings = Settings(tier_medium_size_cap=1.5)
    assert tier_size_mult(ConfidenceTier.MEDIUM, settings) == 1.0


def test_signal_label_a_plus_only_when_every_axis_is_top() -> None:
    settings = Settings()
    top_timing = {"quality": 0.7, "room_r": 2.0}
    assert (
        signal_label(ConfidenceTier.HIGH, 0.7, top_timing, settings) == "A+"
    )
    # Any single degraded axis drops the label to A.
    assert signal_label(ConfidenceTier.HIGH, 0.69, top_timing, settings) == "A"
    assert (
        signal_label(ConfidenceTier.HIGH, 0.7, {"quality": 0.69, "room_r": 2.0}, settings)
        == "A"
    )
    assert (
        signal_label(ConfidenceTier.HIGH, 0.7, {"quality": 0.7, "room_r": 1.9}, settings)
        == "A"
    )
    # Missing data is never invented: A, not A+.
    assert signal_label(ConfidenceTier.HIGH, None, top_timing, settings) == "A"
    assert signal_label(ConfidenceTier.HIGH, 0.7, None, settings) == "A"
    assert signal_label(ConfidenceTier.HIGH, 0.7, {}, settings) == "A"
    # Only HIGH can claim A+ — MEDIUM never, even with top axes.
    assert signal_label(ConfidenceTier.MEDIUM, 0.9, top_timing, settings) == "A"


# ---------------------------------------------------- risk engine tier gate


def _verdicts() -> dict[str, AgentVerdict]:
    return {
        "technical": AgentVerdict(
            agent="technical", model="test", payload={"bias": "long", "conviction": 0.8}
        ),
    }


def _ctx(calibrated: float | None) -> FusionContext:
    return FusionContext(
        fusion=FusionResult(
            side=Side.LONG, direction_score=0.6, raw_confidence=0.6,
            contributions={"technical": 0.6},
        ),
        setup_quality=SetupQuality(score=0.8, components={"regime": 0.8}),
        conflict=ConflictReport(state=ConflictState.ALIGNED, conflicts=[], conflict_score=0.0),
        calibrated_confidence=calibrated,
        regime="trend_up",
        spread_pct=0.05,
    )


def test_low_tier_rejected_with_classified_reason(seeded) -> None:
    engine = RiskEngine(seeded)
    trail: list[dict] = []
    result = engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 2000.0, 10.0, _verdicts(), None,
        fusion_context=_ctx(calibrated=0.2), trail=trail,
    )
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.LOW_TIER.value
    assert "LOW confidence tier" in result.reason
    tier_gates = [t for t in trail if t["gate"] == "tier"]
    assert tier_gates and tier_gates[0]["status"] == "reject"


def test_medium_tier_caps_size(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "TEST/USDT", "15m", Side.LONG, 0.8, 100.0, 1.0, _verdicts(), None,
        fusion_context=_ctx(calibrated=0.5),
    )
    assert isinstance(result, SignalProposal)
    # stop_distance 2.0 -> risk 100 USD -> base size 50.0; MEDIUM cap
    # 0.75 -> 37.5 (exposure floor 5000 not reached: 3750 notional).
    assert result.size == pytest.approx(37.5, rel=1e-6)


def test_high_tier_trades_full_size(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "TEST/USDT", "15m", Side.LONG, 0.8, 100.0, 1.0, _verdicts(), None,
        fusion_context=_ctx(calibrated=0.62),
    )
    assert isinstance(result, SignalProposal)
    assert result.size == pytest.approx(50.0, rel=1e-6)


def test_uncalibrated_is_medium_not_a_block(seeded) -> None:
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "TEST/USDT", "15m", Side.LONG, 0.8, 100.0, 1.0, _verdicts(), None,
        fusion_context=_ctx(calibrated=None),
    )
    assert isinstance(result, SignalProposal)
    assert result.size == pytest.approx(37.5, rel=1e-6)


def test_absolute_exposure_cap_still_applies_after_tier(seeded) -> None:
    # max_exposure 0.5 -> 5000 USD; MEDIUM 37.5 x 100 = 3750 fits. A
    # larger position must still be shrunk by the absolute cap.
    seeded.max_exposure = 0.02
    engine = RiskEngine(seeded)
    result = engine.evaluate(
        "TEST/USDT", "15m", Side.LONG, 0.8, 100.0, 1.0, _verdicts(), None,
        fusion_context=_ctx(calibrated=0.62),  # HIGH -> full tier size
    )
    assert isinstance(result, SignalProposal)
    # base 50.0 (HIGH) -> notional 5000 > allowed 200 -> 2.0
    assert result.size == pytest.approx(2.0, rel=1e-6)


# ------------------------------------------------------- orchestrator stamping


def _settings(**overrides) -> Settings:
    base = dict(
        htf_timeframe="4h",
        snapshot_timeframes=["1h"],
        min_confidence=0.0,
        setup_quality_min=0.0,
        conflict_block_conflicted=False,
        htf_bias_filter_enabled=False,
        room_gate_enabled=False,
        deepseek_api_key=None,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    base.update(overrides)
    return Settings(**base)


def _force_long(orch: Orchestrator, monkeypatch) -> None:
    monkeypatch.setattr(
        orch,
        "_fuse",
        lambda verdicts: FusionResult(side=Side.LONG, direction_score=0.8, raw_confidence=0.8),
    )


def _last_record() -> SignalRecord:
    with session_scope() as session:
        return session.scalars(select(SignalRecord)).first()  # type: ignore[return-value]


def test_pipeline_stamps_tier_and_label_on_proposal(monkeypatch) -> None:
    settings = _settings()
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    row = _last_record()
    # Uncalibrated -> MEDIUM tier -> label A (never A+ without data).
    assert row.fusion["tier"] == "MEDIUM"
    assert row.signal_label == "A"


def test_pipeline_stamps_a_plus_label_when_tier_and_axes_top(monkeypatch) -> None:
    import trading_agent.agents.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "assign_tier", lambda *a, **kw: ConfidenceTier.HIGH)
    monkeypatch.setattr(orch_module, "signal_label", lambda *a, **kw: "A+")
    settings = _settings()
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    row = _last_record()
    assert row.fusion["tier"] == "HIGH"
    assert row.signal_label == "A+"


def test_rejection_record_gets_no_trade_label(monkeypatch) -> None:
    settings = _settings(min_confidence=0.9)  # fused 0.8 -> LOW_CONFIDENCE
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    row = _last_record()
    assert row.final_decision == "rejected"
    assert row.no_trade_reason == NoTradeReason.LOW_CONFIDENCE.value
    assert row.signal_label == NO_TRADE_LABEL


# -------------------------------------------------------------- telegram lines


def _proposal() -> SignalProposal:
    return SignalProposal(
        signal_id="sig-1",
        symbol="XAUUSD",
        timeframe="15m",
        side=Side.LONG,
        confidence=0.72,
        entry=4350.0,
        stop=4300.0,
        target=4450.0,
        size=1.2,
        risk_amount=50.0,
        expected_rr=2.0,
        rationale="r",
        evidence={},
        model="test",
    )


def _record(signal_label: str | None, tier: str | None) -> dict:
    return {
        "signal_id": "sig-1",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "market_snapshot": {},
        "ai_outputs": {},
        "fusion": {"tier": tier} if tier else {},
        "setup_quality": {"score": 0.8, "components": {}},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [],
        "final_decision": "proposal",
        "signal_label": signal_label,
    }


@pytest.fixture
def telegram_settings() -> Settings:
    return Settings(telegram_bot_token="123:abc", telegram_chat_id="987")


def test_proposal_label_line_a_plus_with_tier(telegram_settings) -> None:
    text = TelegramNotifier(telegram_settings).proposal_message(
        _proposal(), record=_record("A+", "HIGH"), proposal_id="pid-9"
    )
    assert "Label : A+ (palier HIGH)" in text


def test_proposal_label_line_a_with_medium_tier(telegram_settings) -> None:
    text = TelegramNotifier(telegram_settings).proposal_message(
        _proposal(), record=_record("A", "MEDIUM"), proposal_id="pid-9"
    )
    assert "Label : A (palier MEDIUM)" in text


def test_proposal_without_label_omits_the_line(telegram_settings) -> None:
    text = TelegramNotifier(telegram_settings).proposal_message(
        _proposal(), record=_record(None, None), proposal_id="pid-9"
    )
    assert "Label :" not in text


def test_rejection_message_carries_no_trade_label(telegram_settings) -> None:
    record = _record(NO_TRADE_LABEL, None)
    record.update(
        final_decision="rejected",
        decision_reason="dollar trop fort pour un LONG",
        no_trade_reason="DXY_FILTER",
    )
    text = TelegramNotifier(telegram_settings).rejection_message(record)
    assert "🚫 SIGNAL REJETÉ XAUUSD (15m)" in text
    assert "Label : NO TRADE" in text
