"""System health + clock + auto-protection (V-MONSTER §76/§77/§78).

One `HealthMonitor` per process scores the loop's own vitals — database
reachability, Telegram delivery, AI availability, data-provider quality,
tick lag and clock skew — into a single 0-100 score. A CRITICAL score
for `health_block_consecutive` evaluations in a row engages the
kill-switch route (`RiskEngine.halt`), which blocks new signals.
Recovery (HEALTHY for `health_recover_consecutive` evaluations)
auto-resets ONLY the halts this monitor engaged — a manual halt is
never touched (spec §4: the trader's kill-switch is sacred).

Checks that do not apply (Telegram disabled, no engine, no cycle yet)
are excluded from the score instead of dragging it down — the score
only ever reflects measurable vitals, and an empty score is 100
(blocking on a vacuum would be dishonest).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import text

from trading_agent.agents.orchestrator import llm_degraded
from trading_agent.config import Settings
from trading_agent.store import db

logger = logging.getLogger(__name__)

# Weights of the six vitals (sum to 1.0; excluded checks renormalize).
COMPONENT_WEIGHTS = {
    "db": 0.25,
    "telegram": 0.15,
    "ai": 0.20,
    "provider": 0.20,
    "tick_lag": 0.10,
    "clock": 0.10,
}


class HealthStatus(StrEnum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class HealthReport:
    """One evaluation: score, status, and every component's verdict."""

    score: float
    status: HealthStatus
    components: dict  # name -> {"included", "ok", "score", "detail"}
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _excluded(detail: str) -> dict:
    return {"included": False, "ok": True, "score": 1.0, "detail": detail}


def _check(name: str, score: float, ok: bool, detail: str) -> dict:
    return {"included": True, "ok": ok, "score": score, "detail": detail}


# ------------------------------------------------------------ the six checks


def check_db() -> dict:
    """Database reachability (a real round-trip, not a cached flag)."""
    if db.engine is None:
        return _excluded("no engine")
    t0 = time.monotonic()
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        ms = (time.monotonic() - t0) * 1000
        return _check("db", 1.0, True, f"ping {ms:.0f}ms")
    except Exception as exc:  # noqa: BLE001 - health must report, not raise
        return _check("db", 0.0, False, f"unreachable: {exc}")


def check_telegram(notifier, s: Settings) -> dict:
    """Phone-channel delivery (disabled = excluded, no traffic = fine)."""
    if not notifier.enabled:
        return _excluded("disabled")
    sends, failures = notifier.sends, notifier.failures
    if sends == 0:
        return _check("telegram", 1.0, True, "no traffic yet")
    rate = 1.0 - failures / sends
    latency_ok = notifier.last_latency_ms <= s.health_telegram_latency_max_ms
    score = rate if latency_ok else min(rate, 0.5)
    detail = (
        f"{sends - failures}/{sends} delivered, "
        f"{notifier.last_latency_ms:.0f}ms last"
    )
    return _check("telegram", score, failures == 0, detail)


def check_ai(verdicts) -> dict:
    """AI availability: real LLM answers vs heuristic fallbacks."""
    if not verdicts:
        return _excluded("no verdicts")
    if llm_degraded(verdicts):
        return _check("ai", 0.2, False, "all agents on fallback")
    real = sum(1 for v in verdicts.values() if getattr(v, "source", "llm") != "fallback")
    frac = real / len(verdicts)
    return _check("ai", max(0.5, frac), True, f"{real}/{len(verdicts)} real answers")


def check_provider(snapshot) -> dict:
    """Data-provider quality of the last cycle (fail-open is visible)."""
    if not snapshot:
        return _excluded("no cycle yet")
    state = snapshot.get("data_quality", "pass")
    score = {"pass": 1.0, "degraded": 0.6, "fail": 0.2}.get(state, 0.5)
    issues = snapshot.get("data_quality_issues") or []
    detail = state + (f" ({'; '.join(issues[:2])})" if issues else "")
    return _check("provider", score, state == "pass", detail)


def check_tick_lag(processing_ms: float, interval_s: float, s: Settings) -> dict:
    """Pipeline time vs the tick interval — the queue/tick lag vitals."""
    budget = interval_s * 1000.0
    if budget <= 0:
        return _excluded("no interval")
    ratio = processing_ms / budget
    if ratio < s.health_tick_lag_ok_ratio:
        score, ok = 1.0, True
    elif ratio < 1.2:
        score, ok = 0.5, False
    else:
        score, ok = 0.1, False
    return _check(
        "tick_lag", score, ok, f"{processing_ms:.0f}ms vs {budget:.0f}ms budget"
    )


def check_clock(snapshot) -> dict:
    """Clock-offset honesty (Phase A flag): market ahead of local clock."""
    if not snapshot:
        return _excluded("no cycle yet")
    issues = snapshot.get("data_quality_issues") or []
    skewed = any("clock" in (i or "").lower() for i in issues)
    if skewed:
        return _check("clock", 0.2, False, "clock skew flagged")
    return _check("clock", 1.0, True, "in tolerance")


# ----------------------------------------------------- aggregation + status


def health_score(components: dict) -> float:
    """Weighted mean over INCLUDED checks only (renormalized weights)."""
    weights = {n: COMPONENT_WEIGHTS[n] for n, c in components.items() if c.get("included")}
    total = sum(weights.values())
    if total <= 0:
        return 100.0  # nothing measurable yet — never block on a vacuum
    return sum(c["score"] * weights[n] for n, c in components.items() if n in weights) / total * 100.0


def classify(score: float, s: Settings) -> HealthStatus:
    if score < s.health_block_score:
        return HealthStatus.CRITICAL
    if score >= s.health_ok_score:
        return HealthStatus.HEALTHY
    return HealthStatus.DEGRADED


def build_report(components: dict, s: Settings) -> HealthReport:
    score = health_score(components)
    return HealthReport(score=round(score, 1), status=classify(score, s), components=components)


def _failing(report: HealthReport, limit: int = 140) -> str:
    names = [
        f"{name}: {c['detail']}"
        for name, c in report.components.items()
        if c.get("included") and not c.get("ok")
    ]
    text = ", ".join(names) or "unknown"
    return text if len(text) <= limit else text[: limit - 3] + "..."


class HealthMonitor:
    """Scores the loop's vitals and drives the health kill-switch.

    `observe()` is called with each tick's measurements; it only
    evaluates when `health_eval_interval_s` has elapsed. CRITICAL
    streaks engage `RiskEngine.halt` (block new signals); HEALTHY
    streaks auto-reset ONLY health-engaged halts.
    """

    def __init__(self, settings: Settings, risk, notifier) -> None:
        self.s = settings
        self.risk = risk
        self.notifier = notifier
        self._last_eval = 0.0
        self._crit_streak = 0
        self._ok_streak = 0
        self.blocked = False
        self.last_report: HealthReport | None = None
        self._pending_event: str | None = None

    def observe(
        self,
        *,
        verdicts=None,
        snapshot=None,
        processing_ms: float = 0.0,
        interval_s: float = 0.0,
    ) -> HealthReport | None:
        """Record this tick's vitals; a report only when due (else None)."""
        now = time.monotonic()
        if now - self._last_eval < self.s.health_eval_interval_s:
            return None
        self._last_eval = now
        components = {
            "db": check_db(),
            "telegram": check_telegram(self.notifier, self.s),
            "ai": check_ai(verdicts),
            "provider": check_provider(snapshot),
            "tick_lag": check_tick_lag(processing_ms, interval_s, self.s),
            "clock": check_clock(snapshot),
        }
        report = build_report(components, self.s)
        self.last_report = report
        self._apply(report)
        return report

    def take_event(self) -> str | None:
        """The pending engage/recover Telegram message, if any (once)."""
        event, self._pending_event = self._pending_event, None
        return event

    def _apply(self, report: HealthReport) -> None:
        if report.status == HealthStatus.CRITICAL:
            self._crit_streak += 1
            self._ok_streak = 0
        elif report.status == HealthStatus.HEALTHY:
            self._crit_streak = 0
            self._ok_streak += 1
        else:
            self._crit_streak = 0
            self._ok_streak = 0

        if not self.blocked and self._crit_streak >= self.s.health_block_consecutive:
            self.blocked = True
            reason = f"system health {report.score:.0f}/100: {_failing(report)}"
            self.risk.halt(reason)
            self._pending_event = (
                f"🚑 Santé système {report.score:.0f}/100 — blocage des "
                f"nouveaux signaux ({_failing(report)}). "
                "reset-halt après revue."
            )
            logger.critical("health monitor blocked new signals: %s", reason)
        elif (
            self.blocked
            and self.s.health_auto_recover
            and self._ok_streak >= self.s.health_recover_consecutive
        ):
            halted, reason = self.risk.is_halted()
            if halted and reason and reason.startswith("system health"):
                self.risk.reset_halt()
                self.blocked = False
                self._pending_event = (
                    f"✅ Santé système rétablie ({report.score:.0f}/100) — "
                    "blocage levé, les signaux reprennent."
                )
                logger.info("health monitor recovered; halt reset")
