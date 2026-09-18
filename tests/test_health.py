"""System health + clock + auto-protection (V-MONSTER §76/§77/§78)."""

from types import SimpleNamespace

import pytest

from trading_agent.config import Settings
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.ops import health
from trading_agent.ops.health import (
    HealthMonitor,
    HealthStatus,
    build_report,
    check_ai,
    check_clock,
    check_db,
    check_provider,
    check_telegram,
    check_tick_lag,
    classify,
    health_score,
)
from trading_agent.risk.engine import RiskEngine


def _fail(name, detail="down"):
    return {"included": True, "ok": False, "score": 0.0, "detail": detail}


def _pass(name, detail="fine"):
    return {"included": True, "ok": True, "score": 1.0, "detail": detail}


# ------------------------------------------------------------------- checks


def test_check_db_reachable_with_engine(fresh_db):
    c = check_db()
    assert c["included"] is True
    assert c["ok"] is True
    assert c["score"] == 1.0


def test_check_db_excluded_without_engine(monkeypatch, fresh_db):
    monkeypatch.setattr(health.db, "engine", None)
    c = check_db()
    assert c["included"] is False


def test_check_telegram_disabled_is_excluded(base_settings):
    notifier = TelegramNotifier(base_settings)  # empty tokens -> disabled
    c = check_telegram(notifier, base_settings)
    assert c["included"] is False


def test_check_telegram_delivery_rate():
    s = Settings(telegram_bot_token="", telegram_chat_id="")
    notifier = SimpleNamespace(enabled=True, sends=10, failures=2, last_latency_ms=100.0)
    c = check_telegram(notifier, s)
    assert c["included"] is True
    assert c["score"] == 0.8
    assert c["ok"] is False
    assert "8/10" in c["detail"]


def test_check_telegram_latency_penalty():
    s = Settings(telegram_bot_token="", telegram_chat_id="")
    notifier = SimpleNamespace(enabled=True, sends=5, failures=0, last_latency_ms=9999.0)
    c = check_telegram(notifier, s)
    assert c["score"] == 0.5  # delivered, but absurdly slow
    assert c["ok"] is True


def test_check_telegram_no_traffic_is_fine():
    s = Settings(telegram_bot_token="", telegram_chat_id="")
    notifier = SimpleNamespace(enabled=True, sends=0, failures=0, last_latency_ms=0.0)
    c = check_telegram(notifier, s)
    assert c["included"] is True
    assert c["score"] == 1.0


def test_check_ai_excluded_without_verdicts():
    assert check_ai({})["included"] is False
    assert check_ai(None)["included"] is False


def test_check_ai_all_fallback_is_bad():
    verdicts = {n: SimpleNamespace(source="fallback") for n in ("a", "b", "c")}
    c = check_ai(verdicts)
    assert c["included"] is True
    assert c["ok"] is False
    assert c["score"] == 0.2


def test_check_ai_mixed_scores_fraction():
    verdicts = {
        "a": SimpleNamespace(source="llm"),
        "b": SimpleNamespace(source="llm"),
        "c": SimpleNamespace(source="fallback"),
    }
    c = check_ai(verdicts)
    assert c["ok"] is True
    assert c["score"] == 2 / 3


def test_check_provider_mapping():
    assert check_provider(None)["included"] is False
    assert check_provider({"data_quality": "pass"})["score"] == 1.0
    assert check_provider({"data_quality": "degraded"})["score"] == 0.6
    assert check_provider({"data_quality": "fail"})["score"] == 0.2
    assert check_provider({"data_quality": "mystery"})["score"] == 0.5


def test_check_tick_lag_bands():
    s = Settings()
    assert check_tick_lag(0.0, 0.0, s)["included"] is False
    assert check_tick_lag(50.0, 1.0, s)["score"] == 1.0  # 5% of budget
    assert check_tick_lag(1000.0, 1.0, s)["score"] == 0.5  # 100% -> warn band
    assert check_tick_lag(1500.0, 1.0, s)["score"] == 0.1  # 150% -> bad


def test_check_clock_skew():
    assert check_clock(None)["included"] is False
    assert check_clock({"data_quality_issues": []})["score"] == 1.0
    skew = {"data_quality_issues": ["market timestamp 12s ahead of local clock"]}
    assert check_clock(skew)["score"] == 0.2
    assert check_clock(skew)["ok"] is False


# ------------------------------------------------------- score + thresholds


def _fail_all(monkeypatch):
    monkeypatch.setattr(health, "check_db", lambda: _fail("db"))
    monkeypatch.setattr(health, "check_telegram", lambda n, s: _fail("telegram"))
    monkeypatch.setattr(health, "check_ai", lambda v: _fail("ai"))
    monkeypatch.setattr(health, "check_provider", lambda sn: _fail("provider"))
    monkeypatch.setattr(health, "check_tick_lag", lambda ms, i, s: _fail("tick_lag"))
    monkeypatch.setattr(health, "check_clock", lambda sn: _fail("clock"))


def _pass_all(monkeypatch):
    monkeypatch.setattr(health, "check_db", lambda: _pass("db"))
    monkeypatch.setattr(health, "check_telegram", lambda n, s: _pass("telegram"))
    monkeypatch.setattr(health, "check_ai", lambda v: _pass("ai"))
    monkeypatch.setattr(health, "check_provider", lambda sn: _pass("provider"))
    monkeypatch.setattr(health, "check_tick_lag", lambda ms, i, s: _pass("tick_lag"))
    monkeypatch.setattr(health, "check_clock", lambda sn: _pass("clock"))


def test_health_score_weighted_mean():
    components = {
        "db": _pass("db"),
        "telegram": _fail("telegram"),
        "ai": {"included": True, "ok": False, "score": 0.2, "detail": "x"},
        "provider": {"included": True, "ok": True, "score": 0.6, "detail": "x"},
        "tick_lag": _pass("tick_lag"),
        "clock": _pass("clock"),
    }
    # 1.0*.25 + 0*.15 + .2*.2 + .6*.2 + 1.0*.1 + 1.0*.1 = .61 -> 61.0
    assert health_score(components) == pytest.approx(61.0)


def test_health_score_renormalizes_excluded():
    components = {
        "db": _pass("db"),
        "telegram": {"included": False, "ok": True, "score": 1.0, "detail": "off"},
        "ai": {"included": True, "ok": False, "score": 0.2, "detail": "x"},
        "provider": {"included": True, "ok": True, "score": 0.6, "detail": "x"},
        "tick_lag": _pass("tick_lag"),
        "clock": _pass("clock"),
    }
    # .61 over remaining weight .85 -> 71.76...
    assert health_score(components) == pytest.approx(0.61 / 0.85 * 100)


def test_health_score_empty_is_perfect():
    assert health_score({}) == 100.0


def test_classify_boundaries():
    s = Settings()  # block 50, ok 80
    assert classify(49.9, s) == HealthStatus.CRITICAL
    assert classify(50.0, s) == HealthStatus.DEGRADED
    assert classify(79.9, s) == HealthStatus.DEGRADED
    assert classify(80.0, s) == HealthStatus.HEALTHY


def test_build_report_status_and_rounding():
    s = Settings()
    components = {n: _pass(n) for n in health.COMPONENT_WEIGHTS}
    report = build_report(components, s)
    assert report.status == HealthStatus.HEALTHY
    assert report.score == 100.0


# ------------------------------------------------------- monitor + kill route


def _monitor(seeded, monkeypatch, settings_tweaks=None):
    if settings_tweaks:
        for key, value in settings_tweaks.items():
            setattr(seeded, key, value)
    monkeypatch.setattr(health, "check_db", lambda: _pass("db"))
    monkeypatch.setattr(health, "check_telegram", lambda n, s: _pass("telegram"))
    monkeypatch.setattr(health, "check_ai", lambda v: _pass("ai"))
    monkeypatch.setattr(health, "check_provider", lambda sn: _pass("provider"))
    monkeypatch.setattr(health, "check_tick_lag", lambda ms, i, s: _pass("tick"))
    monkeypatch.setattr(health, "check_clock", lambda sn: _pass("clock"))
    notifier = TelegramNotifier(seeded)
    return HealthMonitor(seeded, RiskEngine(seeded), notifier)


def test_monitor_blocks_after_critical_streak(seeded, monkeypatch):
    mon = _monitor(
        seeded, monkeypatch, {"health_eval_interval_s": 0.0, "health_block_consecutive": 2}
    )
    # Everything CRITICAL (score 0).
    _fail_all(monkeypatch)
    first = mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)
    assert first.status == HealthStatus.CRITICAL
    assert mon.blocked is False  # streak 1 of 2
    second = mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)
    assert second.status == HealthStatus.CRITICAL
    assert mon.blocked is True
    halted, reason = RiskEngine(seeded).is_halted()
    assert halted is True
    assert reason.startswith("system health")
    assert "blocage" in mon.take_event()


def test_monitor_recovers_health_halt(seeded, monkeypatch):
    mon = _monitor(
        seeded,
        monkeypatch,
        {
            "health_eval_interval_s": 0.0,
            "health_block_consecutive": 1,
            "health_recover_consecutive": 2,
        },
    )
    _fail_all(monkeypatch)
    assert mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)
    assert mon.blocked is True
    # Vitals come back -> HEALTHY streak auto-resets the health halt.
    _pass_all(monkeypatch)
    r1 = mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)
    assert r1.status == HealthStatus.HEALTHY
    assert mon.blocked is True  # streak 1 of 2
    mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)
    assert mon.blocked is False
    assert RiskEngine(seeded).is_halted() == (False, None)
    assert "rétablie" in mon.take_event()


def test_monitor_never_resets_manual_halt(seeded, monkeypatch):
    mon = _monitor(
        seeded,
        monkeypatch,
        {
            "health_eval_interval_s": 0.0,
            "health_block_consecutive": 1,
            "health_recover_consecutive": 1,
        },
    )
    RiskEngine(seeded).halt("manual trader halt")
    # Force blocked state, then healthy: the manual halt must survive.
    mon.blocked = True
    mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)
    halted, reason = RiskEngine(seeded).is_halted()
    assert halted is True
    assert reason == "manual trader halt"
    assert mon.blocked is True  # never claimed the halt


def test_monitor_degrades_reset_critical_streak(seeded, monkeypatch):
    mon = _monitor(
        seeded, monkeypatch, {"health_eval_interval_s": 0.0, "health_block_consecutive": 2}
    )
    _fail_all(monkeypatch)
    mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)  # CRITICAL
    # One DEGRADED evaluation in between breaks the streak: db down only
    # -> 0.25*0 + 0.75*1.0 = 75/100.
    _pass_all(monkeypatch)
    monkeypatch.setattr(health, "check_db", lambda: _fail("db"))
    mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)  # DEGRADED
    _fail_all(monkeypatch)
    mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0)  # CRITICAL
    assert mon.blocked is False  # 2 CRITICALs, but not consecutive


def test_monitor_eval_interval_gates_evaluation(seeded, monkeypatch):
    mon = _monitor(seeded, monkeypatch, {"health_eval_interval_s": 3600.0})
    # First observe evaluates immediately (startup baseline)...
    assert mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0) is not None
    # ...every later call inside the interval is gated.
    assert mon.observe(verdicts={}, snapshot={}, processing_ms=0.0, interval_s=1.0) is None
