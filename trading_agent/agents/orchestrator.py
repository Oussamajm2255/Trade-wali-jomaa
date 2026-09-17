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
from datetime import datetime, timezone
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
from trading_agent.fusion.tier import NO_TRADE_LABEL, assign_tier, signal_label
from trading_agent.fusion.timing import compute_timing
from trading_agent.fusion.types import FusionContext, FusionResult, NoTradeReason
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import (
    AgentVerdict,
    Rejection,
    Side,
    SignalProposal,
    utcnow,
)
from trading_agent.store import actions
from trading_agent.store import opportunity as opportunity_store
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

    def _opportunity_context(
        self,
        symbol: str,
        timeframe: str,
        side: Side,
        snap: MarketSnapshot,
        fusion_context: FusionContext,
        now: datetime | None,
    ) -> dict:
        """Deterministic opportunity identity + dedup verdict (Phase E).

        OPPORTUNITY_ID = direction + structure event + time proximity,
        derived from the canonical snapshot. The verdict is None (allow)
        or {"no_trade_reason", "detail"} (suppress). No anchor never
        blocks — dedup is only possible when the structure is readable.
        """
        anchor = opportunity_store.anchor_from_snapshot(snap, side)
        oid = opportunity_store.opportunity_id(symbol, timeframe, side, anchor)
        verdict = None
        if self.settings.opportunity_dedup_enabled:
            verdict = opportunity_store.dedup_verdict(
                symbol,
                timeframe,
                side,
                anchor,
                price=snap.price,
                setup_score=fusion_context.setup_quality.score,
                settings=self.settings,
                now=now,
            )
        # Phase G (V-MONSTER §42-§49): the tracked opportunity's age
        # feeds the timing lifecycle component. Best-effort lookup —
        # unknown age scores neutral, never blocks.
        age_min = None
        if oid:
            age_min = opportunity_store.opportunity_age_minutes(oid, now)
        return {
            "anchor": anchor,
            "opportunity_id": oid,
            "side": side,
            "verdict": verdict,
            "age_min": age_min,
        }

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
        result, verdicts, entry, gauge, snap, fusion_context, gates, opp = self._run_pipeline(
            symbol, timeframe, signal_id, now
        )
        if signal_id:
            if isinstance(result, (SignalProposal, Rejection)):
                result.signal_id = signal_id
            self._record_signal(
                signal_id, symbol, timeframe, result, snap, verdicts, entry,
                fusion_context, gates, opp, now,
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
        dict | None,
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
                {}, {}, None, None, None, gates, None,
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
                {}, entry, snap.dxy, snap, None, gates, None,
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
                {}, entry, snap.dxy, snap, None, gates, None,
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
                {}, entry, snap.dxy, snap, None, gates, None,
            )
        gates.append({"gate": "cost_control", "status": "pass"})

        # The bias gate's source is the canonical snapshot (fail-closed above).
        htf_bias = snap.biases.get(self.settings.htf_timeframe) if htf_tf else None
        verdicts = self._run_agents(entry, snap.dxy)
        if not verdicts:
            gates.append({"gate": "agents", "status": "reject", "detail": "all analysis agents failed"})
            return (
                Rejection(symbol=symbol, reason="all analysis agents failed"),
                verdicts, entry, snap.dxy, snap, None, gates, None,
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
                verdicts, entry, snap.dxy, snap, None, gates, None,
            )
        gates.append({"gate": "agents", "status": "pass"})

        # Deterministic fusion context (spec §16-§21): setup quality,
        # conflict report and confidence calibration — the risk engine's
        # no-trade gates consume it; it never overrides a hard gate.
        fusion = self._fuse(verdicts)
        fusion_context = build_fusion_context(snap, verdicts, self.settings, htf_tf)

        # Opportunity clustering + dedup (V-MONSTER §31/§40/§41/§52/§53):
        # the same directional setup anchored at the same structure event
        # inside the dedup window is not signalled twice. Deterministic,
        # computed after fusion (side known) and before the risk engine.
        opp = self._opportunity_context(
            symbol, timeframe, fusion.side, snap, fusion_context, now
        )
        if opp["verdict"]:
            gates.append(
                {"gate": "opportunity", "status": "reject", "detail": opp["verdict"]["detail"]}
            )
            return (
                Rejection(
                    symbol=symbol,
                    reason=f"opportunity duplicate: {opp['verdict']['detail']}",
                    no_trade_reason=opp["verdict"]["no_trade_reason"],
                ),
                verdicts, entry, snap.dxy, snap, fusion_context, gates, opp,
            )
        gates.append(
            {"gate": "opportunity", "status": "pass", "detail": opp["opportunity_id"] or "no anchor"}
        )

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
        # Final real-time revalidation (V-MONSTER §56): one fresh tick
        # must still support the proposal after every gate passed. A
        # failure turns the proposal into a recorded rejection — the
        # send never happens (main only sends on SignalProposal).
        if isinstance(result, SignalProposal):
            invalid = self._revalidate_before_send(symbol, result, gates)
            if invalid is not None:
                result = invalid
            else:
                # Timing + actionability gate (V-MONSTER §42-§49/§64):
                # TOO_LATE when the expected lead time is shorter than
                # the human reaction window. The timing payload rides
                # along on the opportunity context so the record and
                # Telegram can render it.
                late = self._timing_before_send(symbol, result, snap, fusion_context, opp, gates, now)
                if late is not None:
                    result = late
        return result, verdicts, entry, snap.dxy, snap, fusion_context, gates, opp

    def _revalidate_before_send(
        self,
        symbol: str,
        proposal: SignalProposal,
        gates: list[dict],
    ) -> Rejection | None:
        """Fresh-tick revalidation after all gates, before the send.

        Checks price drift from the proposal entry, spread and data age
        against the Phase F settings. Fails open: no tick source, no
        data or a fetch error never blocks (honesty, spec §4).
        """
        if not self.settings.final_revalidation_enabled:
            return None
        market = getattr(self, "market", None)
        tick_fn = getattr(market, "tick", None)
        if tick_fn is None:
            gates.append(
                {"gate": "revalidation", "status": "pass", "detail": "no tick source"}
            )
            return None
        try:
            fresh = tick_fn(symbol)
        except Exception as exc:  # noqa: BLE001 - revalidation must never kill the cycle
            logger.warning("final revalidation tick failed for %s: %s", symbol, exc)
            fresh = None
        if not fresh or fresh.get("price") is None:
            gates.append(
                {"gate": "revalidation", "status": "pass", "detail": "tick unavailable"}
            )
            return None

        def reject(detail: str, reason: NoTradeReason) -> Rejection:
            gates.append({"gate": "revalidation", "status": "reject", "detail": detail})
            try:
                actions.audit(
                    "WARNING",
                    "signal_invalidated_pre_send",
                    {"symbol": symbol, "detail": detail},
                )
            except Exception:  # noqa: BLE001 - audit must never kill the cycle
                pass
            return Rejection(
                symbol=symbol,
                reason=f"signal invalidated pre-send: {detail}",
                no_trade_reason=reason.value,
            )

        price = float(fresh["price"])
        drift_pct = (
            abs(price - proposal.entry) / proposal.entry * 100.0 if proposal.entry > 0 else 0.0
        )
        if drift_pct > self.settings.revalidate_max_drift_pct:
            return reject(
                f"price drifted {drift_pct:.4f}% from entry {proposal.entry} to {price}",
                NoTradeReason.SIGNAL_INVALIDATED,
            )
        spread = fresh.get("spread")
        if spread is not None and self.settings.no_trade_max_spread_pct > 0 and price > 0:
            spread_pct = float(spread) / price * 100.0
            if spread_pct > self.settings.no_trade_max_spread_pct:
                return reject(
                    f"spread {spread_pct:.4f}% above maximum "
                    f"{self.settings.no_trade_max_spread_pct}%",
                    NoTradeReason.BAD_SPREAD,
                )
        age = fresh.get("age_s")
        if age is not None and age > self.settings.revalidate_max_age_s:
            return reject(
                f"tick data age {age:.0f}s above maximum {self.settings.revalidate_max_age_s}s",
                NoTradeReason.SIGNAL_INVALIDATED,
            )
        gates.append(
            {"gate": "revalidation", "status": "pass", "detail": f"drift {drift_pct:.4f}%"}
        )
        return None

    def _timing_before_send(
        self,
        symbol: str,
        proposal: SignalProposal,
        snap: MarketSnapshot,
        fusion_context: FusionContext,
        opp: dict | None,
        gates: list[dict],
        now: datetime | None,
    ) -> Rejection | None:
        """Timing + actionability gate (V-MONSTER §42-§49/§64).

        Computes TIMING_QUALITY, lead time, deadline and expected
        execution drift and stamps them on the opportunity context (the
        record and Telegram read them from there). TOO_LATE rejects the
        send when the expected lead time is shorter than the human
        reaction window — an uncomputable pace never blocks (spec §4).
        """
        indicators = (snap.indicators or {}).get(snap.entry_timeframe) or {}
        atr = float(indicators.get("atr_14") or 0.0)
        # Phase I (§63): the EMA of the measured human approval latency
        # overrides the configured reaction budget — the timing model
        # learns the trader's real reaction time. Fail-open on no
        # samples (None -> the configured budget).
        latency_ema = None
        try:
            latency_ema = actions.user_latency_ema(
                span=getattr(self.settings, "user_latency_ema_span", 10) or 10
            )
        except Exception:  # noqa: BLE001 - observation must never kill the cycle
            latency_ema = None
        # The proposal's own side — never the fusion context's (the
        # context re-fuses the real verdicts, which may be NEUTRAL even
        # when the pipeline's fused side produced the proposal).
        timing = compute_timing(
            proposal.side,
            snap,
            self.settings,
            entry=proposal.entry,
            stop=proposal.stop,
            target=proposal.target,
            atr=atr,
            spread_pct=fusion_context.spread_pct,
            opportunity_age_min=(opp or {}).get("age_min"),
            user_reaction_seconds=latency_ema,
            now=now,
        )
        # Phase I (§62): the deadline and chase zone ride on the
        # proposal so the loop's supervision can classify it per tick.
        if isinstance(timing.get("deadline_epoch"), (int, float)):
            proposal.actionability_deadline = datetime.fromtimestamp(
                float(timing["deadline_epoch"]), tz=timezone.utc
            )
        proposal.max_chase = float(timing.get("max_chase") or 0.0)
        if opp is not None:
            opp["timing"] = timing
        # Phase H (V-MONSTER §58/§59/§81): confidence tier + quality
        # label. The label rides on the opportunity context so the
        # record and Telegram render it; the risk engine's LOW gate and
        # MEDIUM size cap use the same tier inputs.
        tier = assign_tier(fusion_context.calibrated_confidence, self.settings)
        label = signal_label(
            tier, fusion_context.setup_quality.score, timing, self.settings
        )
        if opp is not None:
            opp["tier"] = tier.value
            opp["label"] = label
        if not self.settings.timing_gate_enabled or not timing["too_late"]:
            gates.append({"gate": "timing", "status": "pass", "detail": timing["detail"]})
            return None
        detail = (
            f"lead time {timing['lead_time_s']:.0f}s <= reaction "
            f"{timing['reaction_s']:.0f}s"
        )
        gates.append({"gate": "timing", "status": "reject", "detail": detail})
        try:
            actions.audit(
                "WARNING",
                "signal_too_late_pre_send",
                {"symbol": symbol, "detail": detail, "timing": timing},
            )
        except Exception:  # noqa: BLE001 - audit must never kill the cycle
            pass
        return Rejection(
            symbol=symbol,
            reason=f"signal too late pre-send: {detail}",
            no_trade_reason=NoTradeReason.TOO_LATE.value,
        )

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
        opportunity: dict | None,
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
        if opportunity and opportunity.get("opportunity_id"):
            record["opportunity_id"] = opportunity["opportunity_id"]
            record["opportunity_state"] = (
                "TRIGGERED" if isinstance(result, SignalProposal) else "FORMING"
            )
        if fusion_context:
            record["fusion"] = {
                "direction_score": fusion_context.fusion.direction_score,
                "raw_confidence": fusion_context.fusion.raw_confidence,
                "contributions": fusion_context.fusion.contributions,
                "calibrated_confidence": fusion_context.calibrated_confidence,
                "regime": fusion_context.regime,
                "spread_pct": fusion_context.spread_pct,
                # Phase D trigger + Phase F stability ride along so the
                # Telegram lines and the dashboard can render them.
                "trigger": fusion_context.trigger,
                "stability": fusion_context.stability,
            }
            # Phase G timing (V-MONSTER §42-§49/§64) rides along on the
            # opportunity context; proposals and TOO_LATE rejections
            # both carry it.
            if opportunity and opportunity.get("timing"):
                record["fusion"]["timing"] = opportunity["timing"]
            # Phase H (§81): tier and label stamped for the record +
            # Telegram (A+ = top structural + liquidity + timing +
            # statistical bucket; A otherwise; NO TRADE = rejection).
            if opportunity and opportunity.get("tier"):
                record["fusion"]["tier"] = opportunity["tier"]
            record["signal_label"] = (
                opportunity.get("label") if opportunity and opportunity.get("label") else "A"
            )
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
                signal_label=NO_TRADE_LABEL,
            )
        try:
            actions.record_signal(record)
        except Exception as exc:  # noqa: BLE001 - recording must never kill the cycle
            logger.warning("signal recording failed for %s: %s", signal_id, exc)
        # Opportunity lifecycle upsert (V-MONSTER §31): FORMING ->
        # TRIGGERED -> EXPIRED. Observability, best-effort like the record.
        if opportunity and opportunity.get("anchor") and opportunity.get("opportunity_id"):
            try:
                opportunity_store.track_opportunity(
                    opportunity["opportunity_id"],
                    symbol,
                    timeframe,
                    side=opportunity["side"],
                    anchor=opportunity["anchor"],
                    triggered=isinstance(result, SignalProposal),
                    trigger_signal_id=signal_id,
                    settings=self.settings,
                    now=now,
                )
            except Exception as exc:  # noqa: BLE001 - observability must never kill the cycle
                logger.warning("opportunity tracking failed for %s: %s", signal_id, exc)

    def run(self, symbol: str, timeframe: str | None = None) -> SignalProposal | Rejection:
        return self.run_full(symbol, timeframe)[0]
