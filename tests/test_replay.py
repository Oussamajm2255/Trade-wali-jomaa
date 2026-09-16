"""§39 signal replay: reconstruct exactly what the robot knew, with
statistical context that stops at decision time (no look-ahead)."""

import argparse
import json
from datetime import datetime, timedelta, timezone

import pytest

from trading_agent.main import cmd_replay
from trading_agent.store import actions


def _record(signal_id: str, ts: datetime, decision: str = "proposal", **overrides) -> dict:
    record = {
        "signal_id": signal_id,
        "ts": ts,
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "config_version": "v1",
        "prompt_version": "p1",
        "market_snapshot": {
            "symbol": "XAUUSD",
            "timeframe": "15m",
            "data_quality": "good",
            "data_source": "test",
            "last_close": 4350.0,
            "rsi_14": 55.0,
            "adx_14": 28.0,
            "atr_14": 12.5,
            "mtf_biases": {"4h": {"bias": "long"}},
            "regime": {"regime": "trend_up"},
            "structure": {"bos": True},
            "dxy_gauge": {"value": 60, "classification": "Bullish (USD weak)"},
            "dxy_context": {"classification": "Bullish (USD weak)", "trend": "bull"},
            "session_context": {"session": "LONDON"},
            "price": 4350.0,
        },
        "ai_outputs": {
            "technical": {"source": "llm", "payload": {"bias": "long", "conviction": 0.8}},
        },
        "fusion": {"raw_confidence": 0.7, "direction_score": 0.7, "calibrated_confidence": 0.6},
        "setup_quality": {"score": 0.75, "components": {"mtf": 0.8}},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [{"gate": "risk", "status": "pass", "detail": ""}],
        "final_decision": decision,
        "sl": 4300.0,
        "tp": 4450.0,
        "risk_amount": 50.0,
        "size": 1.2,
    }
    record.update(overrides)
    return record


def _args(signal_id: str, as_json: bool = False, context: int = 200, db=None) -> argparse.Namespace:
    return argparse.Namespace(signal_id=signal_id, json=as_json, context=context, db=db)


def test_replay_json_reconstructs_the_full_record(capsys):
    actions.record_signal(_record("sig-1", datetime(2026, 9, 16, 10, 0)))
    cmd_replay(_args("sig-1", as_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["signal_id"] == "sig-1"
    assert payload["decision"] == "proposal"
    assert payload["strategy_version"] == "LEGACY_BASELINE"
    assert payload["sl"] == 4300.0
    assert payload["tp"] == 4450.0
    assert payload["size"] == 1.2
    assert payload["market_snapshot"]["last_close"] == 4350.0
    assert payload["ai_outputs"]["technical"]["payload"]["bias"] == "long"
    assert payload["setup_quality"]["score"] == 0.75
    assert payload["gates"][0]["gate"] == "risk"
    assert payload["contribution"]["supporting"]
    assert payload["contribution"]["invalidation"]
    assert payload["statistical_context"]["resolved_before"] == 0


def test_replay_statistical_context_stops_at_decision_time(capsys):
    now = datetime(2026, 9, 16, 10, 0)
    # A WIN resolved before the decision and a LOSS resolved after it.
    actions.record_signal(
        _record("sig-past", now - timedelta(hours=2), decision="proposal",
                outcome="WIN", r_multiple=1.0)
    )
    actions.record_signal(_record("sig-1", now))
    actions.record_signal(
        _record("sig-future", now + timedelta(hours=2), decision="proposal",
                outcome="LOSS", r_multiple=-1.0)
    )
    cmd_replay(_args("sig-1", as_json=True))
    ctx = json.loads(capsys.readouterr().out)["statistical_context"]
    assert ctx["resolved_before"] == 1
    stats = ctx["stats"]
    assert stats["trades"] == 1
    assert stats["wins"] == 1
    assert stats["expectancy_r"] == 1.0


def test_replay_rejection_record(capsys):
    actions.record_signal(
        _record("sig-r", datetime(2026, 9, 16, 9, 0), decision="rejected",
                decision_reason="dollar trop fort", no_trade_reason="DXY_FILTER",
                sl=None, tp=None, size=None)
    )
    cmd_replay(_args("sig-r", as_json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"] == "rejected"
    assert payload["no_trade_reason"] == "DXY_FILTER"
    assert payload["reason"] == "dollar trop fort"


def test_replay_unknown_signal_exits():
    with pytest.raises(SystemExit) as exc:
        cmd_replay(_args("does-not-exist"))
    assert exc.value.code == 1


def test_replay_rich_output_shows_all_sections(capsys):
    actions.record_signal(_record("sig-1", datetime(2026, 9, 16, 10, 0)))
    cmd_replay(_args("sig-1"))
    out = capsys.readouterr().out
    assert "Signal replay (spec §39)" in out
    assert "trend_up" in out
    assert "LONDON" in out
    assert "Bullish (USD weak)" in out
    assert "Risk gates" in out
    assert "AI outputs" in out
    assert "Soutient :" in out
    assert "Invalidation :" in out
    assert "Statistical context at decision time" in out
