"""Orchestrator: runs the analysis agents, fuses verdicts into a scored
signal, then hands it to the risk engine (which has final say).

Phase 3 (spec §12-§15, §37): agents consume the canonical snapshot only,
every verdict is reliability-tracked, LLM failures are classified and
fall back to labelled heuristics, and too many failed agents block new
proposals (configurable).

Phase 4 (spec §16-§21): fusion returns a signed direction_score plus
raw_confidence and per-agent contributions; the deterministic setup
quality / conflict / calibration context is assembled by the fusion
layer and handed to the risk engine with every candidate.

Phase 5 (spec §22): every cycle — proposal or rejection — is persisted
as one complete signal record: snapshot, AI outputs, fusion, setup
quality, conflicts and the full gate trail."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
from trading_agent.fusion.engine import build_fusion_context, fuse_verdicts
from trading_agent.fusion.types import FusionContext, FusionResult
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import (
    AgentVerdict,
    Rejection,
    SignalProposal,
    utcnow,
)
from trading_agent.store import actions
from trading_agent.versioning import version_stamp

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

    def _fuse(self, verdicts: dict[str, AgentVerdict]) -> FusionResult:
        """Weighted fusion of agent verdicts (spec §16).

        Returns the signed direction_score, raw_confidence and every
        agent's weighted contribution — never one opaque number.
        """
        return fuse_verdicts(verdicts, self.settings)

    def _cost_control_reason(self, entry: dict, snap: MarketSnapshot) -> str | None:
        """Free pre-AI checks (spec §48); a rejection reason when the
        market is clearly invalid, else None.

        Order matters: kill-switch, price, ATR, then the spread ceiling.
        The spread ceiling only applies when the provider supplied a
        spread AND `ai_skip_max_spread_pct` > 0 (0 = disabled).
        """
        if self.risk is not None:
            halted, detail = self.risk.is_halted()
            if halted:
                return f"kill-switch armed — analysis skipped ({detail or 'halted'})"
        price = float(entry.get("last_close") or 0.0)
        if price <= 0:
            return "price is not positive — market invalid"
        atr = float(entry.get("atr_14") or 0.0)
        if atr <= 0:
            return "ATR is not positive — volatility invalid"
        ceiling = self.settings.ai_skip_max_spread_pct
        if ceiling > 0:
            gauge = snap.dxy or {}
            spread = gauge.get("spread") if isinstance(gauge, dict) else None
            if spread:
                spread_pct = float(spread) / price * 100.0
                if spread_pct > ceiling:
                    return (
                        f"spread {spread_pct:.4f}% exceeds ceiling "
                        f"{ceiling:.4f}% — market too expensive to analyze"
                    )
        return None

    def _record_tracks(
        self,
        verdicts: dict[str, AgentVerdict],
        snap: MarketSnapshot,
        signal_id: str | None,
        now: datetime | None,
    ) -> None:
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
                    "ts": now or utcnow(),
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
                    "signal_id": signal_id,
                }
            )
        try:
            actions.record_agent_track(rows)
        except Exception as exc:  # noqa: BLE001 - tracking must never kill the cycle
            logger.warning("agent reliability tracking failed: %s", exc)

    def run_full(
        self,
        symbol: str,
        timeframe: str | None = None,
        now: datetime | None = None,
    ) -> tuple[SignalProposal | Rejection, dict[str, AgentVerdict], dict, dict | None]:
        """Full pipeline; also returns verdicts/snapshot/gauge for reporting.

        Phase 5 (spec §22): every cycle — proposal or rejection — is
        stored as one complete signal record. `now` anchors all
        timestamps (historical replays).
        """
        timeframe = timeframe or self.settings.timeframe
        try:
            signal_id = actions.next_signal_id(symbol, now)
        except Exception as exc:  # noqa: BLE001 - recording must never kill the cycle
            logger.warning("signal id generation failed: %s", exc)
            signal_id = None
        result, verdicts, entry, gauge, snap, fusion_context, gates = self._run_pipeline(
            symbol, timeframe, signal_id, now
        )
        if signal_id:
            if isinstance(result, (SignalProposal, Rejection)):
                result.signal_id = signal_id
            self._record_signal(
                signal_id, symbol, timeframe, result, snap, verdicts, entry,
                fusion_context, gates, now,
            )
        return result, verdicts, entry, gauge

    def _run_pipeline(
        self,
        symbol: str,
        timeframe: str,
        signal_id: str | None,
        now: datetime | None,
    ) -> tuple[
        SignalProposal | Rejection,
        dict[str, AgentVerdict],
        dict,
        dict | None,
        MarketSnapshot | None,
        FusionContext | None,
        list[dict],
    ]:
        """Analysis pipeline; every gate decision lands in the trail."""
        gates: list[dict] = []
        # One coherent snapshot per cycle (spec §3): entry TF + higher TFs,
        # each validated. FAIL stops the cycle before any AI call (§4/§48).
        try:
            snap = build_market_snapshot(
                self.market, symbol, self.settings, timeframe, now=now
            )
        except MarketDataError as exc:
            logger.error("market data unavailable for %s: %s", symbol, exc)
            gates.append(
                {"gate": "market_data", "status": "reject", "detail": str(exc)}
            )
            return (
                Rejection(symbol=symbol, reason=f"market data unavailable: {exc}"),
                {}, {}, None, None, None, gates,
            )
        gates.append({"gate": "market_data", "status": "pass"})
        htf_tf = (
            self.settings.htf_timeframe
            if self.settings.htf_bias_filter_enabled and timeframe != self.settings.htf_timeframe
            else None
        )
        entry = snap.entry_snapshot_for_llm(htf_tf)
        if snap.quality_state == QualityState.FAIL:
            gates.append(
                {"gate": "data_quality", "status": "reject", "detail": "; ".join(snap.quality_issues[:3])}
            )
            return (
                Rejection(symbol=symbol, reason=f"data quality FAIL: {'; '.join(snap.quality_issues[:3])}"),
                {}, entry, snap.dxy, snap, None, gates,
            )
        if snap.degraded and not self.settings.data_quality_allow_degraded:
            gates.append(
                {"gate": "data_quality", "status": "reject", "detail": "; ".join(snap.quality_issues[:3])}
            )
            return (
                Rejection(
                    symbol=symbol,
                    reason=f"data quality DEGRADED (blocked by config): {'; '.join(snap.quality_issues[:3])}",
                ),
                {}, entry, snap.dxy, snap, None, gates,
            )
        gates.append({"gate": "data_quality", "status": "pass"})

        # Cost control (spec §48): a clearly invalid market skips the
        # DeepSeek calls entirely — no verdicts, no API cost.
        cost_reason = self._cost_control_reason(entry, snap)
        if cost_reason:
            logger.warning("cost control: %s — skipping AI calls", cost_reason)
            gates.append({"gate": "cost_control", "status": "reject", "detail": cost_reason})
            return (
                Rejection(symbol=symbol, reason=cost_reason),
                {}, entry, snap.dxy, snap, None, gates,
            )
        gates.append({"gate": "cost_control", "status": "pass"})

        # The bias gate's source is the canonical snapshot (fail-closed above).
        htf_bias = snap.biases.get(self.settings.htf_timeframe) if htf_tf else None
        verdicts = self._run_agents(entry, snap.dxy)
        if not verdicts:
            gates.append({"gate": "agents", "status": "reject", "detail": "all analysis agents failed"})
            return (
                Rejection(symbol=symbol, reason="all analysis agents failed"),
                verdicts, entry, snap.dxy, snap, None, gates,
            )

        # AI reliability tracking (spec §15): every verdict, every cycle.
        self._record_tracks(verdicts, snap, signal_id, now)

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
            gates.append({"gate": "agents", "status": "reject", "detail": block_reason})
            return (
                Rejection(symbol=symbol, reason=block_reason),
                verdicts, entry, snap.dxy, snap, None, gates,
            )
        gates.append({"gate": "agents", "status": "pass"})

        # Deterministic fusion context (spec §16-§21): setup quality,
        # conflict report and confidence calibration — the risk engine's
        # no-trade gates consume it; it never overrides a hard gate.
        fusion = self._fuse(verdicts)
        fusion_context = build_fusion_context(snap, verdicts, self.settings, htf_tf)
        last = snap.candles[timeframe].iloc[-1]
        result = self.risk.evaluate(
            symbol=symbol,
            timeframe=timeframe,
            side=fusion.side,
            confidence=fusion.raw_confidence,
            price=float(last["close"]),
            atr=float(entry["atr_14"]),
            verdicts=verdicts,
            gauge=snap.dxy,
            htf_bias=htf_bias,
            context=snap.context_for_risk(),
            fusion_context=fusion_context,
            trail=gates,
            now=now,
        )
        return result, verdicts, entry, snap.dxy, snap, fusion_context, gates

    def _record_signal(
        self,
        signal_id: str,
        symbol: str,
        timeframe: str,
        result: SignalProposal | Rejection,
        snap: MarketSnapshot | None,
        verdicts: dict[str, AgentVerdict],
        entry: dict,
        fusion_context: FusionContext | None,
        gates: list[dict],
        now: datetime | None,
    ) -> None:
        """Persist one complete signal record (spec §22), best-effort.

        Proposals AND rejections are stored: the robot remembers the
        opportunities it refused, not only the trades it took.
        """
        versions = snap.versions if snap else version_stamp()
        market_snapshot = dict(entry) if snap else {"error": "market data unavailable"}
        if snap:
            market_snapshot["dxy_gauge"] = snap.dxy
            market_snapshot["price"] = snap.price
        record: dict = {
            "signal_id": signal_id,
            "ts": now or utcnow(),
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_version": versions.get("strategy_version", ""),
            "config_version": versions.get("config_version", ""),
            "prompt_version": versions.get("prompt_version", ""),
            "market_snapshot": market_snapshot,
            "ai_outputs": {name: v.model_dump(mode="json") for name, v in verdicts.items()},
            "fusion": {},
            "setup_quality": None,
            "conflicts": None,
            "gates": gates,
            "final_decision": "rejected",
            "decision_reason": None,
            "no_trade_reason": None,
        }
        if fusion_context:
            record["fusion"] = {
                "direction_score": fusion_context.fusion.direction_score,
                "raw_confidence": fusion_context.fusion.raw_confidence,
                "contributions": fusion_context.fusion.contributions,
                "calibrated_confidence": fusion_context.calibrated_confidence,
                "regime": fusion_context.regime,
                "spread_pct": fusion_context.spread_pct,
            }
            record["setup_quality"] = fusion_context.setup_quality.model_dump()
            record["conflicts"] = fusion_context.conflict.model_dump()
        if isinstance(result, SignalProposal):
            record.update(
                final_decision="proposal",
                sl=result.stop,
                tp=result.target,
                risk_amount=result.risk_amount,
                size=result.size,
            )
        else:
            record.update(
                decision_reason=result.reason,
                no_trade_reason=getattr(result, "no_trade_reason", None),
            )
        try:
            actions.record_signal(record)
        except Exception as exc:  # noqa: BLE001 - recording must never kill the cycle
            logger.warning("signal recording failed for %s: %s", signal_id, exc)

    def run(self, symbol: str, timeframe: str | None = None) -> SignalProposal | Rejection:
        return self.run_full(symbol, timeframe)[0]
