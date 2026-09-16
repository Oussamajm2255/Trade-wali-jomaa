"""AI reliability tracking (spec §15): per-agent history, rolling stats.
Analysis-only — weights are never modified from this data."""
from __future__ import annotations

from trading_agent.store import actions
from trading_agent.store.db import session_scope
from trading_agent.store.models import AgentTrack


def _rows(n: int = 3) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "agent": "technical",
                "symbol": "XAUUSD",
                "timeframe": "15m",
                "source": "llm",
                "model": "deepseek-chat",
                "prediction": {"bias": "long", "conviction": 0.7 + i * 0.05},
                "market_regime": "trend_up",
                "confidence": 0.7 + i * 0.05,
                "failure_reason": None,
                "fallback_method": None,
            }
        )
    rows.append(
        {
            "agent": "dxy",
            "symbol": "XAUUSD",
            "timeframe": "15m",
            "source": "fallback",
            "model": "heuristic-fallback",
            "prediction": {"gold_bias": "short", "score": -0.4},
            "market_regime": "range",
            "confidence": 0.4,
            "failure_reason": "TIMEOUT",
            "fallback_method": "heuristic",
        }
    )
    return rows


def test_record_and_aggregate_reliability() -> None:
    actions.record_agent_track(_rows())
    stats = actions.agent_reliability()
    agents = stats["agents"]
    assert set(agents) == {"technical", "dxy"}
    assert agents["technical"]["total"] == 3
    assert agents["technical"]["llm"] == 3
    assert agents["technical"]["failures"] == 0
    assert agents["dxy"]["total"] == 1
    assert agents["dxy"]["fallback"] == 1
    assert agents["dxy"]["failures"] == 1
    # No outcomes yet: accuracy stays None (analysis-only until phase 5).
    assert agents["technical"]["accuracy"] is None
    assert agents["technical"]["evaluated"] == 0


def test_reliability_accuracy_after_outcomes() -> None:
    actions.record_agent_track(_rows())
    with session_scope() as session:
        rows = session.query(AgentTrack).filter(AgentTrack.agent == "technical").all()
        rows[0].actual_outcome = "win"
        rows[0].correct = True
        rows[1].actual_outcome = "loss"
        rows[1].correct = False
    stats = actions.agent_reliability()
    t = stats["agents"]["technical"]
    assert t["evaluated"] == 2
    assert t["accuracy"] == 0.5


def test_reliability_agent_filter() -> None:
    actions.record_agent_track(_rows())
    stats = actions.agent_reliability(agent="dxy")
    assert set(stats["agents"]) == {"dxy"}
    assert stats["agents"]["dxy"]["failures"] == 1


def test_reliability_avg_confidence() -> None:
    actions.record_agent_track(_rows())
    stats = actions.agent_reliability()
    assert stats["agents"]["technical"]["avg_confidence"] == round((0.70 + 0.75 + 0.80) / 3, 4)


def test_empty_tracking_is_safe() -> None:
    actions.record_agent_track([])
    assert actions.agent_reliability()["agents"] == {}
