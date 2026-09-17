"""Signal timing, actionability + execution model (V-MONSTER §42-§49/§64).

The guarantees under test:
- TIMING_QUALITY 0-1 from five deterministic components (trigger
  maturity, speed, remaining room, drift, lifecycle); missing axes
  score a neutral 0.5 and are never fabricated.
- SIGNAL_LEAD_TIME from the speed-adjusted pace, capped at the
  opportunity TTL; ACTIONABILITY_DEADLINE = now + lead.
- EXPECTED_EXECUTION_PRICE/DRIFT over the human reaction window
  (reaction seconds + Telegram latency + spread cost).
- TOO_LATE fires only when the lead time is shorter than the reaction
  window; an uncomputable pace fails open.
- The orchestrator's pre-send TOO_LATE gate aborts the send with a
  recorded rejection + audit event; passes stamp the timing payload on
  the signal record.
- ACTIONABLE_SIGNAL_RATE = proposals / (proposals + TOO_LATE) in the
  dashboard, with the Telegram timing lines rendering the full
  execution model.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.fusion.timing import compute_timing
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, Side, SignalProposal
from trading_agent.store import actions, opportunity as opportunity_store
from trading_agent.store.db import session_scope
from trading_agent.store.models import SignalRecord

from sqlalchemy import select

from test_snapshot import FakeMarket, fresh_gauge


class Snap:
    """Duck-typed snapshot carrying the fields compute_timing reads."""

    def __init__(self, price=100.0, structure=None, speed=None, liquidity=None,
                 timeframe="15m") -> None:
        self.price = price
        self.structure = structure or {}
        self.speed = speed or {}
        self.liquidity = liquidity or {}
        self.entry_timeframe = timeframe


def _timing_kwargs(**overrides) -> dict:
    base = dict(
        entry=100.0,
        stop=95.0,
        target=110.0,
        atr=1.0,
        spread_pct=None,
        opportunity_age_min=None,
        telegram_latency_s=None,
        now=datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------- timing scoring

def test_neutral_side_not_timed() -> None:
    out = compute_timing(Side.NEUTRAL, Snap(), Settings(), **_timing_kwargs())
    assert out["quality"] == 0.5
    assert out["components"] == {}
    assert out["lead_time_s"] is None
    assert out["too_late"] is False
    assert out["deadline_iso"] is None


def test_all_five_components_computed() -> None:
    snap = Snap(
        structure={"bos": [{"type": "BOS_BULLISH", "age": 1}]},
        speed={"state": "NORMAL", "range_per_minute": 1.0},
        liquidity={"nearest_above": {"price": 106.0}},
    )
    out = compute_timing(Side.LONG, snap, Settings(), **_timing_kwargs())
    assert set(out["components"]) == {
        "trigger_maturity", "speed", "remaining_room", "drift", "lifecycle",
    }
    assert all(0.0 <= v <= 1.0 for v in out["components"].values())
    expected = round(sum(out["components"].values()) / 5.0, 4)
    assert out["quality"] == expected


def test_trigger_maturity_fresh_beats_stale() -> None:
    fresh = Snap(structure={"bos": [{"type": "BOS_BULLISH", "age": 1}]})
    stale = Snap(structure={"bos": [{"type": "BOS_BULLISH", "age": 10}]})
    out_fresh = compute_timing(Side.LONG, fresh, Settings(), **_timing_kwargs())
    out_stale = compute_timing(Side.LONG, stale, Settings(), **_timing_kwargs())
    assert out_fresh["components"]["trigger_maturity"] == round(1 - 1 / 12, 4)
    assert out_stale["components"]["trigger_maturity"] == round(1 - 10 / 12, 4)
    assert out_fresh["components"]["trigger_maturity"] > out_stale["components"]["trigger_maturity"]


def test_trigger_maturity_no_events_is_neutral() -> None:
    out = compute_timing(Side.LONG, Snap(), Settings(), **_timing_kwargs())
    assert out["components"]["trigger_maturity"] == 0.5


def test_speed_component_mapping() -> None:
    for state, expected in (("SLOW", 1.0), ("NORMAL", 0.8), ("FAST", 0.5), ("EXTREME", 0.2)):
        snap = Snap(speed={"state": state})
        out = compute_timing(Side.LONG, snap, Settings(), **_timing_kwargs())
        assert out["components"]["speed"] == expected


def test_remaining_room_normalized_and_neutral_without_liquidity() -> None:
    # room_r = 6 / (1 * 2) = 3 R -> min(1, 3 / (2 * 1)) = 1.0
    with_level = Snap(liquidity={"nearest_above": {"price": 106.0}})
    out = compute_timing(Side.LONG, with_level, Settings(), **_timing_kwargs())
    assert out["components"]["remaining_room"] == 1.0
    # No opposing level mapped: honest neutral, never fabricated.
    out2 = compute_timing(Side.LONG, Snap(), Settings(), **_timing_kwargs())
    assert out2["components"]["remaining_room"] == 0.5


def test_lifecycle_component_consumes_ttl() -> None:
    out = compute_timing(Side.LONG, Snap(), Settings(), **_timing_kwargs(opportunity_age_min=360.0))
    assert out["components"]["lifecycle"] == 0.5  # 360 of 720 minutes
    out2 = compute_timing(Side.LONG, Snap(), Settings(), **_timing_kwargs())
    assert out2["components"]["lifecycle"] == 0.5  # unknown age: neutral


def test_drift_component_counts_reaction_and_spread() -> None:
    snap = Snap(speed={"state": "NORMAL", "range_per_minute": 1.0})
    out = compute_timing(
        Side.LONG, snap, Settings(),
        **_timing_kwargs(entry=100.0, stop=95.0, spread_pct=0.1),
    )
    # reaction 123 s -> 2.05 min move at 1.0/min + 0.1% of 100 = 0.1
    assert out["expected_drift"] == round(2.05 + 0.1, 8)
    # stop distance 5 -> drift component 1 - 2.15/5 = 0.57
    assert out["components"]["drift"] == round(1 - 2.15 / 5, 4)
    # LONG expects execution above entry.
    assert out["expected_price"] == round(100.0 + 2.15, 8)


# -------------------------------------------------------- lead + deadline

def test_lead_time_and_deadline_from_pace() -> None:
    snap = Snap(speed={"state": "NORMAL", "range_per_minute": 1.0})
    now = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
    out = compute_timing(
        Side.LONG, snap, Settings(),
        **_timing_kwargs(entry=100.0, target=160.0, now=now),
    )
    assert out["lead_time_s"] == 3600.0  # 60 / 1.0 * 60 s
    assert out["too_late"] is False
    assert out["reaction_s"] == 123.0  # 120 + 3 latency
    assert out["deadline_epoch"] == (now + timedelta(seconds=3600)).timestamp()
    assert out["deadline_iso"] == "2026-09-17 11:00:00 UTC"


def test_too_late_when_lead_shorter_than_reaction() -> None:
    snap = Snap(speed={"state": "FAST", "range_per_minute": 100.0})
    out = compute_timing(
        Side.LONG, snap, Settings(),
        **_timing_kwargs(entry=100.0, target=110.0),
    )
    # FAST pace multiplier 1.5 -> 150/min -> 10 / 150 * 60 s
    assert out["lead_time_s"] == 4.0
    assert out["too_late"] is True


def test_lead_capped_at_opportunity_ttl() -> None:
    snap = Snap(speed={"state": "NORMAL", "range_per_minute": 0.001})
    out = compute_timing(
        Side.LONG, snap, Settings(),
        **_timing_kwargs(entry=100.0, target=1000.0),
    )
    assert out["lead_time_s"] == 720 * 60.0


def test_pace_falls_back_to_atr_per_minute() -> None:
    snap = Snap(speed={"state": "NORMAL"}, timeframe="15m")  # no rates in speed
    out = compute_timing(
        Side.SHORT, snap, Settings(),
        **_timing_kwargs(entry=100.0, target=40.0, atr=15.0),
    )
    # base = 15 / 15 min = 1.0/min -> 60 / 1.0 * 60 s
    assert out["lead_time_s"] == 3600.0
    # SHORT expects execution below entry.
    assert out["expected_price"] < 100.0


def test_unknown_pace_fails_open() -> None:
    out = compute_timing(
        Side.LONG, Snap(), Settings(),
        **_timing_kwargs(entry=100.0, atr=0.0),
    )
    assert out["pace_per_minute"] is None
    assert out["lead_time_s"] is None
    assert out["too_late"] is False
    assert out["deadline_iso"] is None


def test_execution_zone_and_max_chase() -> None:
    out = compute_timing(Side.LONG, Snap(), Settings(), **_timing_kwargs(atr=1.0))
    assert out["max_chase"] == 0.5  # 0.5 ATR
    assert out["execution_zone"] == [100.0, 100.5]
    out2 = compute_timing(Side.SHORT, Snap(), Settings(), **_timing_kwargs(atr=1.0))
    assert out2["execution_zone"] == [99.5, 100.0]
    # chase disabled: degenerate zone.
    settings = Settings(max_chase_atr_mult=0.0)
    out3 = compute_timing(Side.LONG, Snap(), settings, **_timing_kwargs(atr=1.0))
    assert out3["max_chase"] == 0.0
    assert out3["execution_zone"] == [100.0, 100.0]


# -------------------------------------------------- opportunity age helper

def test_opportunity_age_minutes_unknown_when_missing() -> None:
    assert opportunity_store.opportunity_age_minutes("XAUUSD:15m:LONG:bos:nope") is None


def test_opportunity_age_minutes_tracks_first_seen() -> None:
    oid = "XAUUSD:15m:LONG:bos:t1"
    now = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)
    opportunity_store.track_opportunity(
        oid, "XAUUSD", "15m", Side.LONG, "bos:t1",
        triggered=False, trigger_signal_id=None, settings=Settings(), now=now,
    )
    assert opportunity_store.opportunity_age_minutes(oid, now + timedelta(minutes=5)) == 5.0


# ---------------------------------------------------- orchestrator TOO_LATE

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
    from trading_agent.fusion.types import FusionResult

    monkeypatch.setattr(
        orch,
        "_fuse",
        lambda verdicts: FusionResult(side=Side.LONG, direction_score=0.8, raw_confidence=0.8),
    )


def _fake_timing(too_late: bool) -> dict:
    return {
        "quality": 0.3,
        "components": {
            "trigger_maturity": 0.5, "speed": 0.5, "remaining_room": 0.5,
            "drift": 0.0, "lifecycle": 0.0,
        },
        "speed_state": "FAST",
        "pace_per_minute": 50.0,
        "lead_time_s": 6.0,
        "reaction_s": 123.0,
        "deadline_epoch": None,
        "deadline_iso": None,
        "expected_price": 101.0,
        "expected_drift": 1.0,
        "max_chase": 0.5,
        "execution_zone": [100.0, 100.5],
        "too_late": too_late,
        "detail": "timing 0.30",
    }


def _patch_timing(monkeypatch, too_late: bool) -> None:
    import trading_agent.agents.orchestrator as orch_module

    monkeypatch.setattr(
        orch_module, "compute_timing", lambda *a, **kw: _fake_timing(too_late)
    )


def test_timing_gate_too_late_aborts_send(monkeypatch) -> None:
    _patch_timing(monkeypatch, too_late=True)
    orch = Orchestrator(_settings(), FakeMarket(gauge=fresh_gauge()), RiskEngine(_settings()))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == "TOO_LATE"
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        assert row.final_decision == "rejected"
        assert row.no_trade_reason == "TOO_LATE"
        gates = [g for g in (row.gates or []) if g["gate"] == "timing"]
        assert gates and gates[0]["status"] == "reject"
        # The timing payload rides the record for forensics.
        assert row.fusion["timing"]["too_late"] is True
    events = [a.event for a in actions.recent_audit()]
    assert "signal_too_late_pre_send" in events


def test_timing_gate_pass_stamps_record(monkeypatch) -> None:
    _patch_timing(monkeypatch, too_late=False)
    settings = _settings()
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        gates = [g for g in (row.gates or []) if g["gate"] == "timing"]
        assert gates and gates[0]["status"] == "pass"
        assert row.fusion["timing"]["too_late"] is False
        assert row.fusion["timing"]["quality"] == 0.3


def test_timing_gate_disabled_never_blocks(monkeypatch) -> None:
    _patch_timing(monkeypatch, too_late=True)
    settings = _settings(timing_gate_enabled=False)
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        gates = [g for g in (row.gates or []) if g["gate"] == "timing"]
        assert gates and gates[0]["status"] == "pass"


def test_real_pipeline_timing_runs_without_blocking(monkeypatch) -> None:
    """End-to-end: real compute_timing on synthetic data never blocks a
    normal-speed proposal (lead ~60 min >> reaction window)."""
    settings = _settings()
    orch = Orchestrator(settings, FakeMarket(gauge=fresh_gauge()), RiskEngine(settings))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        timing = row.fusion.get("timing") or {}
        assert timing.get("quality") is not None
        assert timing.get("too_late") is False
        assert timing.get("lead_time_s") > 123.0


# ------------------------------------------------------------ telegram

def _proposal() -> SignalProposal:
    return SignalProposal(
        id="pid-1",
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


def test_timing_lines_render_execution_model() -> None:
    record = {
        "signal_id": "sig-1",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "market_snapshot": {"data_quality": "good", "price": 4350.0},
        "ai_outputs": {},
        "fusion": {
            "direction_score": 0.7,
            "raw_confidence": 0.7,
            "timing": {
                "quality": 0.65,
                "speed_state": "NORMAL",
                "lead_time_s": 3600.0,
                "reaction_s": 123.0,
                "deadline_iso": "2026-09-17 10:30:00 UTC",
                "expected_price": 4352.15,
                "expected_drift": 2.15,
                "max_chase": 6.25,
                "execution_zone": [4350.0, 4356.25],
                "too_late": False,
            },
        },
        "setup_quality": {"score": 0.75},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [{"gate": "timing", "status": "pass"}],
        "final_decision": "proposal",
    }
    notifier = TelegramNotifier(Settings(telegram_bot_token="123:abc", telegram_chat_id="987"))
    text = notifier.proposal_message(_proposal(), record=record)
    assert "Timing : 0.65 (bon)" in text
    assert "Zone d'exécution : 4,350.00 – 4,356.25" in text
    assert "Chasse max : 6.25" in text
    assert "Prix attendu : 4,352.15 (dérive 2.15)" in text
    assert "Valable jusqu'à : 2026-09-17 10:30:00 UTC" in text
    assert "timing ✓" in text


def test_timing_lines_omitted_without_payload() -> None:
    record = {
        "signal_id": "sig-1",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "market_snapshot": {"data_quality": "good", "price": 4350.0},
        "ai_outputs": {},
        "fusion": {"direction_score": 0.7, "raw_confidence": 0.7},
        "setup_quality": {"score": 0.75},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [{"gate": "timing", "status": "pass"}],
        "final_decision": "proposal",
    }
    notifier = TelegramNotifier(Settings(telegram_bot_token="123:abc", telegram_chat_id="987"))
    text = notifier.proposal_message(_proposal(), record=record)
    assert "Timing :" not in text
    assert "Valable jusqu'à" not in text


# ---------------------------------------------------- dashboard KPI

def _dash_signal(signal_id, ts, decision="proposal", no_trade_reason=None) -> dict:
    return {
        "signal_id": signal_id,
        "ts": ts,
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "config_version": "v1",
        "prompt_version": "p1",
        "market_snapshot": {
            "symbol": "XAUUSD", "timeframe": "15m", "data_quality": "good",
            "price": 4350.0,
            "regime": {"regime": "trend_up"},
            "session_context": {"session": "LONDON"},
            "alignment": {"label": "aligned"},
            "dxy_gauge": {"value": 60, "classification": "Bullish (USD weak)"},
            "mtf_biases": {"4h": {"bias": "long"}},
        },
        "ai_outputs": {},
        "fusion": {"raw_confidence": 0.7, "direction_score": 0.7},
        "setup_quality": {"score": 0.7},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [{"gate": "risk", "status": "pass", "detail": ""}],
        "final_decision": decision,
        "no_trade_reason": no_trade_reason,
        "outcome": None,
        "r_multiple": None,
    }


def test_actionable_rate_aggregation() -> None:
    from trading_agent.dashboard.report import collect

    today = datetime.now(timezone.utc).replace(microsecond=0)
    actions.record_signal(_dash_signal("p1", today - timedelta(minutes=5)))
    actions.record_signal(_dash_signal("p2", today - timedelta(minutes=4)))
    actions.record_signal(
        _dash_signal("t1", today - timedelta(minutes=3), decision="rejected",
                     no_trade_reason="TOO_LATE")
    )
    actions.record_signal(
        _dash_signal("t2", today - timedelta(minutes=2), decision="rejected",
                     no_trade_reason="TOO_LATE")
    )
    actions.record_signal(
        _dash_signal("r1", today - timedelta(minutes=1), decision="rejected",
                     no_trade_reason="DXY_FILTER")
    )
    with session_scope() as session:
        data = collect(session)
    # 2 proposals / (2 proposals + 2 too-late) = 50%; other reasons excluded.
    assert data["today"]["actionable_rate"] == 0.5


def test_actionable_rate_none_without_activity() -> None:
    from trading_agent.dashboard.report import collect

    with session_scope() as session:
        assert collect(session)["today"]["actionable_rate"] is None


def test_actionable_rate_card_in_html() -> None:
    from trading_agent.dashboard.report import build_dashboard_html

    today = datetime.now(timezone.utc).replace(microsecond=0)
    actions.record_signal(_dash_signal("p1", today - timedelta(minutes=5)))
    actions.record_signal(
        _dash_signal("t1", today - timedelta(minutes=3), decision="rejected",
                     no_trade_reason="TOO_LATE")
    )
    with session_scope() as session:
        html = build_dashboard_html(session)
    assert "Signaux actionnables" in html
    assert "50.0%" in html
