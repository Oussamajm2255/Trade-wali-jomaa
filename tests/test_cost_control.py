"""Cost control (spec §48): skip the DeepSeek calls when the market is
clearly invalid — kill-switch, non-positive price/ATR, spread ceiling.
No network, no LLM: `_run_agents` is monkeypatched to prove the AI
layer is never reached on a skip."""

from __future__ import annotations

import pytest

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.data.quality import QualityState
from trading_agent.schema.types import Rejection


def make_orchestrator(risk=None, **overrides) -> Orchestrator:
    settings = Settings(
        weight_technical=0.45, weight_regime=0.35, weight_sentiment=0.20,
        side_threshold=0.25, **overrides,
    )
    return Orchestrator(settings, market=None, risk=risk)  # type: ignore[arg-type]


class _HaltedRisk:
    def is_halted(self) -> tuple[bool, str | None]:
        return True, "manual halt"


class _CleanRisk:
    def is_halted(self) -> tuple[bool, str | None]:
        return False, None


class _FakeSnap:
    quality_state = QualityState.PASS
    degraded = False
    quality_issues: list = []
    biases: dict = {}

    def __init__(self, entry=None, dxy=None) -> None:
        self._entry = entry if entry is not None else {"last_close": 4350.0, "atr_14": 12.5}
        self.dxy = dxy if dxy is not None else {}

    def entry_snapshot_for_llm(self, htf_timeframe=None) -> dict:
        return dict(self._entry)


def _run(orch: Orchestrator, monkeypatch, snap: _FakeSnap, *, agents_result=None):
    monkeypatch.setattr(
        "trading_agent.agents.orchestrator.build_market_snapshot",
        lambda *a, **k: snap,
    )
    if agents_result is None:
        monkeypatch.setattr(
            orch, "_run_agents",
            lambda *a, **k: pytest.fail("AI agents must not run on a skip path"),
        )
    else:
        monkeypatch.setattr(orch, "_run_agents", lambda *a, **k: agents_result)
    return orch._run_pipeline("XAUUSD", "15m", None, None)


def _cost_gate(gates) -> dict | None:
    return next((g for g in gates if g["gate"] == "cost_control"), None)


# ------------------------------------------------------------- kill-switch


def test_kill_switch_skips_ai(monkeypatch):
    orch = make_orchestrator(risk=_HaltedRisk())
    result, verdicts, entry, gauge, snap, _, gates, _opp = _run(orch, monkeypatch, _FakeSnap())
    assert isinstance(result, Rejection)
    assert "kill-switch" in result.reason
    assert verdicts == {}  # zero AI outputs — nothing was called
    gate = _cost_gate(gates)
    assert gate == {"gate": "cost_control", "status": "reject", "detail": result.reason}
    assert snap is not None  # snapshot still returned for the record


# ------------------------------------------------------------- price / ATR


def test_zero_price_skips_ai(monkeypatch):
    orch = make_orchestrator()
    snap = _FakeSnap(entry={"last_close": 0.0, "atr_14": 12.5})
    result, verdicts, _, _, _, _, gates, _ = _run(orch, monkeypatch, snap)
    assert isinstance(result, Rejection)
    assert "price" in result.reason
    assert verdicts == {}
    assert _cost_gate(gates)["status"] == "reject"


def test_zero_atr_skips_ai(monkeypatch):
    orch = make_orchestrator()
    snap = _FakeSnap(entry={"last_close": 4350.0, "atr_14": 0.0})
    result, verdicts, _, _, _, _, gates, _ = _run(orch, monkeypatch, snap)
    assert isinstance(result, Rejection)
    assert "ATR" in result.reason
    assert verdicts == {}
    assert _cost_gate(gates)["status"] == "reject"


# ------------------------------------------------------------ spread ceiling


def test_spread_over_ceiling_skips_ai(monkeypatch):
    orch = make_orchestrator(ai_skip_max_spread_pct=0.05)
    snap = _FakeSnap(dxy={"value": 60, "spread": 3.0})  # 3/4350 = 0.069% > 0.05
    result, verdicts, _, _, _, _, gates, _ = _run(orch, monkeypatch, snap)
    assert isinstance(result, Rejection)
    assert "spread" in result.reason and "0.0690" in result.reason
    assert verdicts == {}
    assert _cost_gate(gates)["status"] == "reject"


def test_spread_within_ceiling_passes(monkeypatch):
    orch = make_orchestrator(ai_skip_max_spread_pct=0.05)
    snap = _FakeSnap(dxy={"value": 60, "spread": 1.0})  # 0.023% < 0.05
    result, verdicts, _, _, _, _, gates, _ = _run(orch, monkeypatch, snap, agents_result={})
    assert _cost_gate(gates) == {"gate": "cost_control", "status": "pass"}
    # agents ran (and all failed) — the skip never fired.
    assert isinstance(result, Rejection)
    assert "analysis agents failed" in result.reason


def test_ceiling_zero_disables_spread_check(monkeypatch):
    orch = make_orchestrator()  # default ai_skip_max_spread_pct = 0.0
    snap = _FakeSnap(dxy={"value": 60, "spread": 50.0})
    result, _, _, _, _, _, gates, _ = _run(orch, monkeypatch, snap, agents_result={})
    assert _cost_gate(gates)["status"] == "pass"
    assert isinstance(result, Rejection)  # blocked later, not by spread


# --------------------------------------------------------------- clean market


def test_clean_market_appends_pass_gate(monkeypatch):
    orch = make_orchestrator(risk=_CleanRisk())
    result, _, _, _, _, _, gates, _ = _run(orch, monkeypatch, _FakeSnap(), agents_result={})
    assert _cost_gate(gates) == {"gate": "cost_control", "status": "pass"}
    assert isinstance(result, Rejection)  # all-agents-failed stub, gate passed


def test_helper_returns_none_when_risk_is_none(monkeypatch):
    # make_orchestrator keeps risk=None (the unit-test default).
    orch = make_orchestrator()
    assert orch._cost_control_reason({"last_close": 4350.0, "atr_14": 12.5}, _FakeSnap()) is None


def test_gate_trail_keeps_market_and_data_quality_passes(monkeypatch):
    orch = make_orchestrator(ai_skip_max_spread_pct=0.05)
    snap = _FakeSnap(dxy={"value": 60, "spread": 3.0})
    _, _, _, _, _, _, gates, _ = _run(orch, monkeypatch, snap)
    trail = [g["gate"] for g in gates]
    assert trail == ["market_data", "data_quality", "cost_control"]
    assert gates[0]["status"] == "pass"
    assert gates[1]["status"] == "pass"
