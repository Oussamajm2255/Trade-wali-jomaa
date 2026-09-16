"""Orchestrator: runs the analysis agents, fuses verdicts into a scored
signal, then hands it to the risk engine (which has final say).

Phase 3 (spec §12-§15, §37): agents consume the canonical snapshot only,
every verdict is reliability-tracked, LLM failures are classified and
fall back to labelled heuristics, and too many failed agents block new
proposals (configurable)."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from trading_agent.agents.base import LLMClient
from trading_agent.agents.dxy import DxyContextAgent
from trading_agent.agents.regime import RegimeAgent
from trading_agent.agents.sentiment import SentimentAgent
from trading_agent.agents.technical import TechnicalAgent
from trading_agent.config import Settings
from trading_agent.data.market import MarketData, MarketDataError
from trading_agent.data.quality import QualityState
from trading_agent.data.snapshot import MarketSnapshot, build_market_snapshot
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import (
    AgentVerdict,
    Bias,
    Regime,
    Rejection,
    Side,
    SignalProposal,
)
from trading_agent.store import actions

logger = logging.getLogger(__name__)


def llm_degraded(verdicts: dict[str, AgentVerdict]) -> bool:
    """True when every available verdict came from the heuristic fallback
    (LLM unreachable — e.g. empty DeepSeek balance)."""
    return bool(verdicts) and all(v.source == "fallback" for v in verdicts.values())


def failure_block_reason(failed: list[str], settings: Settings) -> str | None:
    """Reason to block proposals when too many agents failed, else None.

    Spec §37: 0 failed = normal, 1 = degraded warning, N >= block_min =
    block new proposals. Pure function so the policy is unit-testable.
    """
    if not failed or not settings.agent_failure_block_enabled:
        return None
    if len(failed) >= settings.agent_failure_block_min:
        return (
            f"{len(failed)} analysis agent(s) failed (DEGRADED_MODE): "
            f"{', '.join(sorted(failed))} — new proposals blocked"
        )
    return None


class Orchestrator:
    def __init__(self, settings: Settings, market: MarketData, risk: RiskEngine) -> None:
        self.settings = settings
        self.market = market
        self.risk = risk
        self.llm = LLMClient(settings)
        self.technical = TechnicalAgent(self.llm, settings)
        self.sentiment = SentimentAgent(self.llm, settings)
        self.regime = RegimeAgent(self.llm, settings)
        self.dxy = DxyContextAgent(self.llm, settings)

    def _run_agents(self, snapshot: dict, gauge: dict | None) -> dict[str, AgentVerdict]:
        # Gold cycles have a deterministic DXY context block -> the DXY
        # context agent (§14). Other symbols keep the legacy sentiment agent.
        if snapshot.get("dxy_context"):
            context_agent: Callable[[], AgentVerdict] = lambda: self.dxy.analyze(snapshot, gauge)
            context_name = "dxy"
        else:
            context_agent = lambda: self.sentiment.analyze(gauge, snapshot)
            context_name = "sentiment"
        tasks: dict[str, Callable[[], AgentVerdict]] = {
            "technical": lambda: self.technical.analyze(snapshot),
            "regime": lambda: self.regime.analyze(snapshot),
            context_name: context_agent,
        }
        verdicts: dict[str, AgentVerdict] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    verdicts[name] = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate agent crashes
                    logger.error("agent %s crashed: %s", name, exc)
        return verdicts

    def _fuse(self, verdicts: dict[str, AgentVerdict]) -> tuple[Side, float]:
        """Weighted fusion of agent verdicts into (side, confidence)."""
        score = 0.0
        for name, verdict in verdicts.items():
            payload = verdict.payload
            if name == "technical":
                bias = Bias(payload["bias"])
                sign = 1.0 if bias == Bias.LONG else -1.0 if bias == Bias.SHORT else 0.0
                score += self.settings.weight_technical * sign * payload["conviction"]
            elif name == "sentiment":
                score += self.settings.weight_sentiment * payload["score"]
            elif name == "dxy":
                # Signed score agrees with gold_bias by schema; the bias
                # gate is belt-and-braces so a wrong sign cannot count.
                bias = Bias(payload["gold_bias"])
                sign = 1.0 if bias == Bias.LONG else -1.0 if bias == Bias.SHORT else 0.0
                score += self.settings.weight_sentiment * sign * abs(payload["score"])
            else:  # regime
                # Prefer the explicit trend_direction (§13); fall back to
                # the legacy regime enum mapping for older payloads.
                direction = payload.get("trend_direction")
                if direction == "up":
                    sign = 1.0
                elif direction == "down":
                    sign = -1.0
                elif direction == "flat":
                    sign = 0.0
                else:
                    regime = Regime(payload["regime"])
                    sign = (
                        1.0
                        if regime == Regime.TRENDING_UP
                        else -1.0
                        if regime == Regime.TRENDING_DOWN
                        else 0.0
                    )
                score += self.settings.weight_regime * sign * payload["trend_strength"]
        threshold = self.settings.side_threshold
        if score >= threshold:
            side = Side.LONG
        elif score <= -threshold:
            side = Side.SHORT
        else:
            side = Side.NEUTRAL
        return side, min(1.0, abs(score))

    def _record_tracks(self, verdicts: dict[str, AgentVerdict], snap: MarketSnapshot) -> None:
        """Reliability tracking (spec §15) — analysis-only, best-effort."""
        regime_label = (snap.regimes.get(snap.entry_timeframe) or {}).get("regime")
        rows = []
        for name, verdict in verdicts.items():
            payload = verdict.payload
            confidence = (
                payload.get("confidence")
                if payload.get("confidence") is not None
                else payload.get("conviction")
                if payload.get("conviction") is not None
                else abs(payload.get("score")) if payload.get("score") is not None else None
            )
            rows.append(
                {
                    "agent": name,
                    "symbol": snap.symbol,
                    "timeframe": snap.entry_timeframe,
                    "source": verdict.source,
                    "model": verdict.model,
                    "prediction": payload,
                    "market_regime": regime_label,
                    "confidence": confidence,
                    "failure_reason": verdict.failure_reason,
                    "fallback_method": "heuristic" if verdict.source == "fallback" else None,
                }
            )
        try:
            actions.record_agent_track(rows)
        except Exception as exc:  # noqa: BLE001 - tracking must never kill the cycle
            logger.warning("agent reliability tracking failed: %s", exc)

    def run_full(
        self, symbol: str, timeframe: str | None = None
    ) -> tuple[SignalProposal | Rejection, dict[str, AgentVerdict], dict, dict | None]:
        """Full pipeline; also returns verdicts/snapshot/gauge for reporting."""
        timeframe = timeframe or self.settings.timeframe
        # One coherent snapshot per cycle (spec §3): entry TF + higher TFs,
        # each validated. FAIL stops the cycle before any AI call (§4/§48).
        try:
            snap = build_market_snapshot(self.market, symbol, self.settings, timeframe)
        except MarketDataError as exc:
            logger.error("market data unavailable for %s: %s", symbol, exc)
            return Rejection(symbol=symbol, reason=f"market data unavailable: {exc}"), {}, {}, None
        htf_tf = (
            self.settings.htf_timeframe
            if self.settings.htf_bias_filter_enabled and timeframe != self.settings.htf_timeframe
            else None
        )
        entry = snap.entry_snapshot_for_llm(htf_tf)
        if snap.quality_state == QualityState.FAIL:
            return (
                Rejection(symbol=symbol, reason=f"data quality FAIL: {'; '.join(snap.quality_issues[:3])}"),
                {},
                entry,
                snap.dxy,
            )
        if snap.degraded and not self.settings.data_quality_allow_degraded:
            return (
                Rejection(
                    symbol=symbol,
                    reason=f"data quality DEGRADED (blocked by config): {'; '.join(snap.quality_issues[:3])}",
                ),
                {},
                entry,
                snap.dxy,
            )

        # The bias gate's source is the canonical snapshot (fail-closed above).
        htf_bias = snap.biases.get(self.settings.htf_timeframe) if htf_tf else None
        verdicts = self._run_agents(entry, snap.dxy)
        if not verdicts:
            return Rejection(symbol=symbol, reason="all analysis agents failed"), verdicts, entry, snap.dxy

        # AI reliability tracking (spec §15): every verdict, every cycle.
        self._record_tracks(verdicts, snap)

        # Fallback isolation (spec §37): too many failed agents = no new
        # proposals. One failure degrades loudly but keeps the cycle.
        failed = [name for name, v in verdicts.items() if v.failure_reason]
        block_reason = failure_block_reason(failed, self.settings)
        if failed:
            if block_reason:
                actions.audit("WARNING", "agents_blocked_proposals", {"failed": failed})
            else:
                logger.warning("degraded: agent(s) fell back: %s", failed)
                actions.audit("WARNING", "agent_degraded", {"failed": failed})
        if block_reason:
            return Rejection(symbol=symbol, reason=block_reason), verdicts, entry, snap.dxy

        side, confidence = self._fuse(verdicts)
        last = snap.candles[timeframe].iloc[-1]
        result = self.risk.evaluate(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            confidence=confidence,
            price=float(last["close"]),
            atr=float(entry["atr_14"]),
            verdicts=verdicts,
            gauge=snap.dxy,
            htf_bias=htf_bias,
            context=snap.context_for_risk(),
        )
        return result, verdicts, entry, snap.dxy

    def run(self, symbol: str, timeframe: str | None = None) -> SignalProposal | Rejection:
        return self.run_full(symbol, timeframe)[0]
