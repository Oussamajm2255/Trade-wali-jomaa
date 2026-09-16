"""Statistical quality (spec §29-§32): feature extraction, regime
breakdowns, conditional expectancy, minimum samples and the opt-in
STATISTICAL_QUALITY risk gate."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import trading_agent.store.db as db
from trading_agent.analytics.stats import (
    breakdown,
    candidate_features,
    conditional_expectancy,
    dxy_class,
    resolved_signals,
    signal_features,
    statistical_quality,
)
from trading_agent.fusion.types import (
    ConflictReport,
    ConflictState,
    FusionContext,
    FusionResult,
    NoTradeReason,
    SetupQuality,
)
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import AgentVerdict, Side, SignalProposal, Rejection
from trading_agent.store.models import Position, SignalRecord


def _signal(
    signal_id: str,
    ts: datetime,
    fusion: dict | None = None,
    snapshot: dict | None = None,
    outcome: str | None = "WIN",
    r: float | None = 1.0,
    symbol: str = "XAUUSD",
    proposal_id: str | None = None,
) -> SignalRecord:
    return SignalRecord(
        signal_id=signal_id,
        ts=ts,
        symbol=symbol,
        timeframe="15m",
        strategy_version="test",
        config_version="test",
        prompt_version="test",
        market_snapshot=snapshot or {},
        ai_outputs={},
        fusion=fusion or {},
        gates=[],
        final_decision="proposal" if outcome else "rejected",
        outcome=outcome,
        r_multiple=r,
        proposal_id=proposal_id,
    )


def _verdicts() -> dict[str, AgentVerdict]:
    return {
        "technical": AgentVerdict(
            agent="technical", model="test", payload={"bias": "long", "conviction": 0.8}
        )
    }


def _fusion_ctx(regime: str | None = "trend_up", quality: float = 0.8) -> FusionContext:
    return FusionContext(
        fusion=FusionResult(
            side=Side.LONG, direction_score=0.6, raw_confidence=0.6,
            contributions={"technical": 0.6},
        ),
        setup_quality=SetupQuality(score=quality, components={"regime": quality}),
        conflict=ConflictReport(state=ConflictState.ALIGNED, conflicts=[], conflict_score=0.0),
        regime=regime,
    )


# --- feature extraction ----------------------------------------------------


def test_signal_features_extract_all_dimensions() -> None:
    row = _signal(
        "SF1",
        datetime(2026, 1, 1),
        fusion={"direction_score": -0.7, "regime": "trend_down"},
        snapshot={
            "session_context": {"session": "LONDON"},
            "alignment": {"label": "aligned"},
            "structure": {"bos": [1], "choch": [1], "fvgs": [1], "sweeps": [1]},
            "dxy_context": {"classification": "Bearish (USD strong)"},
        },
    )
    assert signal_features(row) == {
        "side": "short",
        "regime": "TREND_DOWN",
        "session": "LONDON",
        "alignment": "aligned",
        "structure": ["BOS", "CHoCH", "FVG", "LIQUIDITY_SWEEP"],
        "dxy": "supportive",  # a strong dollar supports a gold short
    }


def test_dxy_class_semantics() -> None:
    weak = {"dxy_context": {"classification": "Bullish (USD weak)"}}
    strong = {"dxy_context": {"classification": "Bearish (USD strong)"}}
    neutral = {"dxy_context": {"classification": "Neutral"}}
    assert dxy_class("long", weak) == "supportive"
    assert dxy_class("long", strong) == "contradictory"
    assert dxy_class("short", weak) == "contradictory"
    assert dxy_class("short", strong) == "supportive"
    assert dxy_class("long", neutral) == "neutral"
    assert dxy_class("long", {}) == "unknown"
    assert dxy_class(None, weak) == "unknown"


# --- §29 breakdowns / §30 conditional expectancy --------------------------


def test_breakdown_groups_by_dimension() -> None:
    rows = [
        _signal("SB1", datetime(2026, 1, 1), fusion={"direction_score": 0.5},
                snapshot={"session_context": {"session": "LONDON"}}, outcome="WIN", r=1.0),
        _signal("SB2", datetime(2026, 1, 2), fusion={"direction_score": 0.5},
                snapshot={"session_context": {"session": "LONDON"}}, outcome="LOSS", r=-1.0),
        _signal("SB3", datetime(2026, 1, 3), fusion={"direction_score": 0.5},
                snapshot={"session_context": {"session": "NEW_YORK"}}, outcome="WIN", r=2.0),
    ]
    groups = breakdown(rows, "session")
    assert set(groups) == {"LONDON", "NEW_YORK"}
    assert groups["LONDON"].trades == 2
    assert groups["LONDON"].expectancy_r == 0.0
    assert groups["NEW_YORK"].expectancy_r == 2.0
    # missing dimension values fall into UNKNOWN, never crash
    unknown = breakdown(rows, "alignment")
    assert unknown["UNKNOWN"].trades == 3


def test_conditional_expectancy_filters_conditions() -> None:
    rows = [
        _signal("SC1", datetime(2026, 1, 1), fusion={"direction_score": 0.5, "regime": "trend_up"},
                outcome="WIN", r=2.0),
        _signal("SC2", datetime(2026, 1, 2), fusion={"direction_score": 0.5, "regime": "range"},
                outcome="LOSS", r=-1.0),
        _signal("SC3", datetime(2026, 1, 3), fusion={"direction_score": -0.5, "regime": "trend_up"},
                outcome="LOSS", r=-1.0),
    ]
    stats = conditional_expectancy(rows, {"side": "long", "regime": "TREND_UP"})
    assert stats.trades == 1
    assert stats.expectancy_r == 2.0


def test_structure_conditions_require_all_labels() -> None:
    row = _signal(
        "SS1", datetime(2026, 1, 1),
        fusion={"direction_score": 0.5},
        snapshot={"structure": {"bos": [1], "fvgs": [1]}},
    )
    rows = [row]
    assert conditional_expectancy(rows, {"structure": ["BOS", "FVG"]}).trades == 1
    assert conditional_expectancy(rows, {"structure": ["BOS", "CHoCH"]}).trades == 0


# --- §31 minimum samples / §32 quality ------------------------------------


def test_resolved_signals_excludes_unresolved_and_future_rows() -> None:
    boundary = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
    with db.SessionLocal() as session:
        session.add_all([
            _signal("SR1", datetime(2026, 1, 5, 9, 0), outcome="WIN", r=1.0),
            _signal("SR2", datetime(2026, 1, 5, 10, 0), outcome="LOSS", r=-1.0),
            _signal("SR3", datetime(2026, 1, 5, 11, 0), outcome=None, r=None),
            _signal("SR4", datetime(2026, 1, 5, 13, 0), outcome="WIN", r=1.0),  # future
        ])
        session.commit()
        rows = resolved_signals(session, before=boundary)
    assert [r.signal_id for r in rows] == ["SR1", "SR2"]


def test_statistical_quality_below_min_sample_is_unknown_not_sufficient() -> None:
    rows = [
        _signal("SQ1", datetime(2026, 1, 1), fusion={"direction_score": 0.5, "regime": "trend_up"},
                outcome="WIN", r=1.0),
        _signal("SQ2", datetime(2026, 1, 2), fusion={"direction_score": 0.5, "regime": "trend_up"},
                outcome="LOSS", r=-1.0),
    ]
    features = candidate_features("long", "trend_up", None)
    quality = statistical_quality(rows, features, min_sample=5)
    assert quality.sample_size == 2
    assert quality.known
    assert not quality.sufficient
    assert quality.historical_expectancy == 0.0


def test_statistical_quality_computes_mfe_mae_from_positions() -> None:
    with db.SessionLocal() as session:
        session.add(_signal(
            "SM1", datetime(2026, 1, 1),
            fusion={"direction_score": 0.5, "regime": "trend_up"},
            snapshot={"dxy_context": {"classification": "Bullish (USD weak)"}},
            outcome="WIN", r=2.0, proposal_id="p1",
        ))
        session.add(Position(
            proposal_id="p1", symbol="XAUUSD", side="long", size=1.0,
            entry=3000.0, stop=2990.0, target=3020.0, status="closed",
            mfe_price=3015.0, mae_price=2995.0,
        ))
        session.commit()
        rows = resolved_signals(session)
        positions = {p.proposal_id: p for p in session.scalars(select(Position))}
        features = candidate_features(
            "long", "trend_up", {"classification": "Bullish (USD weak)"}
        )
        quality = statistical_quality(rows, features, min_sample=1, positions=positions)
    assert quality.sample_size == 1
    assert quality.sufficient
    # MFE: (3015-3000)/10 = 1.5 R; MAE: (2995-3000)/10 = -0.5 R
    assert quality.mfe_r == pytest.approx(1.5)
    assert quality.mae_r == pytest.approx(-0.5)


# --- the opt-in §32 gate in the risk engine --------------------------------


def _seed_similar_signals(outcome: str, r: float, n: int = 2) -> None:
    with db.SessionLocal() as session:
        for i in range(n):
            session.add(_signal(
                f"SG{i}",
                datetime(2026, 1, i + 1, tzinfo=timezone.utc),
                fusion={"direction_score": 0.6, "regime": "trend_up"},
                snapshot={"dxy_context": {"classification": "Bullish (USD weak)"}},
                outcome=outcome, r=r,
            ))
        session.commit()


def _evaluate(seeded, trail: list[dict]):
    engine = RiskEngine(seeded)
    return engine.evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 3000.0, 10.0, _verdicts(),
        {"kind": "dxy", "value": 80.0, "classification": "Bullish (USD weak)"},
        fusion_context=_fusion_ctx(regime="trend_up"),
        trail=trail,
    )


def test_statistical_quality_gate_disabled_by_default(seeded) -> None:
    _seed_similar_signals("LOSS", -1.0)
    trail: list[dict] = []
    result = _evaluate(seeded, trail)
    assert isinstance(result, SignalProposal)
    gates = {g["gate"]: g for g in trail}
    assert gates["statistical_quality"]["status"] == "pass"


def test_statistical_quality_blocks_negative_expectancy(seeded) -> None:
    seeded.statistical_quality_enabled = True
    seeded.min_sample_for_statistics = 2
    seeded.statistical_quality_min_expectancy = 0.0
    _seed_similar_signals("LOSS", -1.0)
    trail: list[dict] = []
    result = _evaluate(seeded, trail)
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.STATISTICAL_QUALITY.value
    assert "expectancy" in result.reason
    gates = {g["gate"]: g for g in trail}
    assert gates["statistical_quality"]["status"] == "reject"


def test_statistical_quality_insufficient_sample_never_blocks(seeded) -> None:
    seeded.statistical_quality_enabled = True
    seeded.min_sample_for_statistics = 5  # only 2 similar signals seeded
    seeded.statistical_quality_min_expectancy = 0.0
    _seed_similar_signals("LOSS", -1.0)
    trail: list[dict] = []
    result = _evaluate(seeded, trail)
    assert isinstance(result, SignalProposal)
    gates = {g["gate"]: g for g in trail}
    assert gates["statistical_quality"]["status"] == "unknown"


def test_statistical_quality_positive_expectancy_passes(seeded) -> None:
    seeded.statistical_quality_enabled = True
    seeded.min_sample_for_statistics = 2
    seeded.statistical_quality_min_expectancy = 0.0
    _seed_similar_signals("WIN", 1.5)
    trail: list[dict] = []
    result = _evaluate(seeded, trail)
    assert isinstance(result, SignalProposal)
    gates = {g["gate"]: g for g in trail}
    assert gates["statistical_quality"]["status"] == "pass"


def test_statistical_quality_ignores_signals_resolved_after_now(seeded) -> None:
    seeded.statistical_quality_enabled = True
    seeded.min_sample_for_statistics = 2
    seeded.statistical_quality_min_expectancy = 0.0
    # Future-dated losses must not be visible: the boundary is `now`.
    _seed_similar_signals("LOSS", -1.0)
    with db.SessionLocal() as session:
        session.add(_signal(
            "SGF", datetime(2030, 1, 1, tzinfo=timezone.utc),
            fusion={"direction_score": 0.6, "regime": "trend_up"},
            snapshot={"dxy_context": {"classification": "Bullish (USD weak)"}},
            outcome="LOSS", r=-1.0,
        ))
        session.commit()
    trail: list[dict] = []
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)
    result = RiskEngine(seeded).evaluate(
        "XAUUSD", "15m", Side.LONG, 0.8, 3000.0, 10.0, _verdicts(),
        {"kind": "dxy", "value": 80.0, "classification": "Bullish (USD weak)"},
        fusion_context=_fusion_ctx(regime="trend_up"),
        trail=trail,
        now=now,
    )
    # Only the two 2026 rows are visible -> sufficient sample -> reject.
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.STATISTICAL_QUALITY.value


def test_statistical_quality_is_symbol_scoped(seeded) -> None:
    seeded.statistical_quality_enabled = True
    seeded.min_sample_for_statistics = 1
    seeded.statistical_quality_min_expectancy = 0.0
    with db.SessionLocal() as session:
        session.add(_signal(
            "SGS", datetime(2026, 1, 1, tzinfo=timezone.utc),
            fusion={"direction_score": 0.6, "regime": "trend_up"},
            snapshot={"dxy_context": {"classification": "Bullish (USD weak)"}},
            outcome="LOSS", r=-1.0, symbol="ETH/USDT",
        ))
        session.commit()
    trail: list[dict] = []
    result = _evaluate(seeded, trail)
    # The only seeded loss belongs to another symbol: XAUUSD sample is 0.
    assert isinstance(result, SignalProposal)
    gates = {g["gate"]: g for g in trail}
    assert gates["statistical_quality"]["status"] == "unknown"
