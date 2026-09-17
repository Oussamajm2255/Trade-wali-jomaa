"""Risk engine — the deterministic layer with final authority.

The LLM proposes; this module disposes. Every gate is hard-coded,
unit-tested and audit-logged. Kill-switch state is persisted, so a halt
survives restarts.

Phase 4 (spec §20): the no-trade gates — LOW_CONFIDENCE, LOW_SETUP_
QUALITY, deterministic conflicts, HIGH_VOLATILITY, BAD_SPREAD and
STATISTICAL_EDGE_UNKNOWN — consume the fusion context and refuse
proposals with a classified no_trade_reason.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_agent.config import Settings
from trading_agent.data.calendar import blocking_events
from trading_agent.data.shock import ShockState
from trading_agent.fusion.room import compute_room
from trading_agent.fusion.types import ConflictState, FusionContext, NoTradeReason
from trading_agent.schema.types import AgentVerdict, Rejection, Side, SignalProposal
from trading_agent.store.db import session_scope
from trading_agent.store.models import AuditLog, Position, RiskState

logger = logging.getLogger(__name__)

MIN_NOTIONAL = 5.0  # refuse dust-sized paper positions (USD)


def _as_utc(dt: datetime) -> datetime:
    """Normalise a possibly-naive SQLite datetime to aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class RiskEngine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    # ------------------------------------------------------------------ state

    def _get_or_create_state(self, session: Session, now: datetime | None = None) -> RiskState:
        state = session.get(RiskState, 1)
        if state is None:
            equity = self.s.paper_starting_equity
            today = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
            state = RiskState(
                id=1,
                equity=equity,
                start_of_day_equity=equity,
                day=today,
                peak_equity=equity,
                halted=False,
            )
            session.add(state)
            session.flush()
        self._roll_day(state, now)
        return state

    @staticmethod
    def _roll_day(state: RiskState, now: datetime | None = None) -> None:
        today = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
        if state.day != today:
            state.start_of_day_equity = state.equity
            state.day = today

    def _halt(self, session: Session, state: RiskState, reason: str) -> None:
        if state.halted:
            return
        state.halted = True
        state.halt_reason = reason
        session.add(
            AuditLog(level="CRITICAL", event="kill_switch", detail={"reason": reason})
        )
        logger.critical("KILL-SWITCH ENGAGED: %s", reason)

    def _enforce_limits(self, session: Session, state: RiskState) -> None:
        if state.halted:
            return
        day_pnl = state.equity - state.start_of_day_equity
        if state.start_of_day_equity > 0 and day_pnl <= -self.s.daily_loss_limit * state.start_of_day_equity:
            self._halt(session, state, f"daily loss limit breached: {day_pnl:.2f}")
            return
        if state.peak_equity > 0 and state.equity <= state.peak_equity * (1 - self.s.max_drawdown):
            self._halt(
                session,
                state,
                f"max drawdown breached: equity {state.equity:.2f} vs peak {state.peak_equity:.2f}",
            )
            return
        if state.equity <= 0:
            self._halt(session, state, "equity depleted")
            return

    # ------------------------------------------------------------------ API

    def is_halted(self) -> tuple[bool, str | None]:
        with session_scope() as session:
            state = self._get_or_create_state(session)
            return state.halted, state.halt_reason

    def halt(self, reason: str) -> None:
        with session_scope() as session:
            self._halt(session, self._get_or_create_state(session), reason)

    def reset_halt(self) -> None:
        with session_scope() as session:
            state = self._get_or_create_state(session)
            state.halted = False
            state.halt_reason = None
            session.add(
                AuditLog(level="WARNING", event="kill_switch_reset", detail={})
            )

    def realize_pnl(self, session: Session, pnl: float, now: datetime | None = None) -> None:
        """Apply realised PnL to equity, update peak, enforce limits.

        Runs inside the caller's session/transaction — the caller owns
        commit/rollback (the paper broker calls this within session_scope).
        `now` lets historical replays roll the daily-loss day correctly.
        """
        state = self._get_or_create_state(session, now)
        state.equity = round(state.equity + pnl, 8)
        state.peak_equity = max(state.peak_equity, state.equity)
        self._enforce_limits(session, state)

    def set_equity(self, equity: float) -> None:
        """Replace equity with the broker's authoritative value (live mode).

        Daily-loss / drawdown limits are enforced on the REAL broker equity,
        not on local approximations. Peak only ever ratchets up.
        """
        with session_scope() as session:
            state = self._get_or_create_state(session)
            state.equity = round(equity, 8)
            state.peak_equity = max(state.peak_equity, state.equity)
            self._enforce_limits(session, state)

    # ------------------------------------------------------------------ gates

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        side: Side,
        confidence: float,
        price: float,
        atr: float,
        verdicts: dict[str, AgentVerdict],
        gauge: dict | None,
        htf_bias: dict | None = None,
        context: dict | None = None,
        fusion_context: FusionContext | None = None,
        trail: list[dict] | None = None,
        now: datetime | None = None,
    ) -> SignalProposal | Rejection:
        """Apply every hard gate; return a proposal or a logged rejection.

        `trail` (optional) collects every gate decision as
        {"gate", "status", "detail"} so the signal record (spec §22) can
        store the full gate trail. `now` anchors day rollover and
        timestamps for historical replays (backtesting).
        """

        def mark(gate: str, status: str, detail: str = "") -> None:
            if trail is not None:
                trail.append({"gate": gate, "status": status, "detail": detail})

        with session_scope() as session:
            state = self._get_or_create_state(session, now)
            self._enforce_limits(session, state)

            if state.halted:
                reason = f"kill-switch engaged: {state.halt_reason}"
                mark("kill_switch", "reject", reason)
                return Rejection(symbol=symbol, reason=reason)
            if side == Side.NEUTRAL:
                mark("side", "reject", "fused signal is neutral")
                return Rejection(symbol=symbol, reason="fused signal is neutral")
            if confidence < self.s.min_confidence:
                mark(
                    "confidence",
                    "reject",
                    f"confidence {confidence:.2f} below minimum {self.s.min_confidence}",
                )
                return Rejection(
                    symbol=symbol,
                    reason=f"confidence {confidence:.2f} below minimum {self.s.min_confidence}",
                    no_trade_reason=NoTradeReason.LOW_CONFIDENCE.value,
                )

            # --- News gate (spec §43), opt-in. Events come from a real
            # calendar provider only; a HIGH-importance USD event within
            # news_block_minutes refuses new entries with NEWS_RISK.
            if self.s.news_filter_enabled:
                news = (context or {}).get("news_context") or []
                blocking = blocking_events(
                    news, self.s.news_min_importance, self.s.news_block_minutes
                )
                if blocking:
                    detail = "; ".join(
                        f"{e['event']} in {e['minutes_to_event']}m ({e['importance']})"
                        for e in blocking[:3]
                    )
                    mark("news", "reject", detail)
                    return Rejection(
                        symbol=symbol,
                        reason=f"news blackout: {detail}",
                        no_trade_reason=NoTradeReason.NEWS_RISK.value,
                    )
                mark("news", "pass", f"{len(news)} event(s) in window")

            # --- Shock gate (spec §44). SHOCK blocks new entries (or
            # applies the persisted cooldown); VOLATILITY_EXPANSION is a
            # warning in the trail, never a block by itself.
            if self.s.shock_enabled:
                shock = (context or {}).get("shock_context") or {}
                shock_state = shock.get("state") or ShockState.NORMAL
                now_ts = now or datetime.now(timezone.utc)
                if shock_state == ShockState.SHOCK:
                    state.last_shock_ts = now_ts  # cooldown anchor, persisted
                    detail = shock.get("detail", "shock")
                    if self.s.shock_block_new_entries:
                        mark("shock", "reject", detail)
                        return Rejection(
                            symbol=symbol,
                            reason=f"market shock: {detail}",
                            no_trade_reason=NoTradeReason.SHOCK.value,
                        )
                    mark("shock", "warning", detail)
                elif (
                    state.last_shock_ts is not None
                    and self.s.shock_cooldown_minutes > 0
                    and _as_utc(state.last_shock_ts)
                    + timedelta(minutes=self.s.shock_cooldown_minutes)
                    >= now_ts
                ):
                    elapsed = int((now_ts - _as_utc(state.last_shock_ts)).total_seconds() // 60)
                    detail = (
                        f"shock cooldown: {self.s.shock_cooldown_minutes - elapsed}m remaining"
                    )
                    mark("shock", "reject", detail)
                    return Rejection(
                        symbol=symbol,
                        reason=f"market shock cooldown: {detail}",
                        no_trade_reason=NoTradeReason.SHOCK.value,
                    )
                elif shock_state == ShockState.VOLATILITY_EXPANSION:
                    mark("shock", "warning", shock.get("detail", "volatility expansion"))
                else:
                    mark("shock", "pass", shock.get("detail", "normal volatility"))

            # --- No-trade gates (spec §20), fusion-layer inputs. ---
            no_trade = self._no_trade_rejection(symbol, fusion_context)
            if no_trade:
                mark("no_trade", "reject", f"{no_trade.no_trade_reason or 'no_trade'}: {no_trade.reason}")
                return no_trade

            # --- Room-to-target gate (V-MONSTER §29, Phase C). ---
            # The distance to the opposing liquidity pool, after spread
            # and slippage, must leave at least room_min_rr R. No mapped
            # opposing level never blocks (data honesty, spec §4).
            room = self._room_gate(symbol, side, price, atr, context, fusion_context)
            if room:
                mark("room", "reject", room.reason)
                return room
            mark("room", "pass")

            # --- Statistical quality gate (spec §32), opt-in. ---
            # Pipeline order (spec): AI SIGNAL -> SETUP QUALITY ->
            # STATISTICAL QUALITY -> HARD RISK ENGINE. Unknown history
            # (insufficient sample) never blocks (spec §31); a sufficient
            # sample whose historical expectancy is below the configured
            # floor refuses the proposal with STATISTICAL_QUALITY.
            if self.s.statistical_quality_enabled:
                sq = self._statistical_quality(
                    session, symbol, side, fusion_context, gauge, now
                )
                if sq.sufficient and sq.historical_expectancy is not None and (
                    sq.historical_expectancy
                    < self.s.statistical_quality_min_expectancy
                ):
                    reason = (
                        f"statistical quality: {sq.sample_size} similar signals, "
                        f"historical expectancy {sq.historical_expectancy:.4f} R below "
                        f"floor {self.s.statistical_quality_min_expectancy:.4f}"
                    )
                    mark("statistical_quality", "reject", reason)
                    return Rejection(
                        symbol=symbol,
                        reason=reason,
                        no_trade_reason=NoTradeReason.STATISTICAL_QUALITY.value,
                    )
                status = "pass" if sq.sufficient else "unknown"
                mark(
                    "statistical_quality",
                    status,
                    f"sample {sq.sample_size}/{sq.min_sample}, "
                    f"expectancy {sq.historical_expectancy}",
                )
            else:
                mark("statistical_quality", "pass", "gate disabled")

            # DXY concurrency hard gate (gold): the dollar must agree with
            # the trade direction or the signal is refused — no exceptions.
            if self.s.dxy_filter_enabled and gauge and gauge.get("kind") == "dxy":
                value = float(gauge["value"])
                if side == Side.LONG and value < self.s.dxy_long_min:
                    reason = (
                        f"DXY concurrency: gauge {value:.0f} not weak-dollar "
                        f"(need >= {self.s.dxy_long_min:.0f}) for a LONG"
                    )
                    mark("dxy_concurrency", "reject", reason)
                    return Rejection(symbol=symbol, reason=reason)
                if side == Side.SHORT and value > self.s.dxy_short_max:
                    reason = (
                        f"DXY concurrency: gauge {value:.0f} not strong-dollar "
                        f"(need <= {self.s.dxy_short_max:.0f}) for a SHORT"
                    )
                    mark("dxy_concurrency", "reject", reason)
                    return Rejection(symbol=symbol, reason=reason)
            mark("dxy_concurrency", "pass")
            # HTF bias hard gate (multi-timeframe): the 4h trend must agree
            # with the entry direction; a choppy HTF blocks both sides.
            if self.s.htf_bias_filter_enabled and htf_bias:
                bias = htf_bias.get("bias")
                if side == Side.LONG and bias != "bull":
                    reason = (
                        f"HTF bias: {bias} on {self.s.htf_timeframe} blocks LONG "
                        f"({htf_bias.get('detail', '')})"
                    )
                    mark("htf_bias", "reject", reason)
                    return Rejection(symbol=symbol, reason=reason)
                if side == Side.SHORT and bias != "bear":
                    reason = (
                        f"HTF bias: {bias} on {self.s.htf_timeframe} blocks SHORT "
                        f"({htf_bias.get('detail', '')})"
                    )
                    mark("htf_bias", "reject", reason)
                    return Rejection(symbol=symbol, reason=reason)
            mark("htf_bias", "pass")
            open_positions = list(session.scalars(select(Position).where(Position.status == "open")))
            if any(p.symbol == symbol for p in open_positions):
                mark("positions", "reject", "position already open for this symbol")
                return Rejection(symbol=symbol, reason="position already open for this symbol")
            if len(open_positions) >= self.s.max_positions:
                mark("positions", "reject", f"max positions reached ({self.s.max_positions})")
                return Rejection(symbol=symbol, reason=f"max positions reached ({self.s.max_positions})")
            mark("positions", "pass")
            if atr <= 0 or price <= 0:
                mark("sizing", "reject", "invalid price/ATR for sizing")
                return Rejection(symbol=symbol, reason="invalid price/ATR for sizing")

            exposure = sum(p.size * p.entry for p in open_positions)
            stop_distance = atr * self.s.atr_stop_mult
            risk_amount = state.equity * self.s.risk_per_trade
            size = risk_amount / stop_distance

            # Exposure cap: shrink size if it would breach portfolio exposure.
            allowed = self.s.max_exposure * state.equity - exposure
            if size * price > allowed:
                size = max(0.0, allowed / price)
            if size * price < MIN_NOTIONAL:
                mark(
                    "notional",
                    "reject",
                    f"position notional {size * price:.2f} below floor {MIN_NOTIONAL}",
                )
                return Rejection(
                    symbol=symbol,
                    reason=f"position notional {size * price:.2f} below floor {MIN_NOTIONAL}",
                )

            if side == Side.LONG:
                stop = price - stop_distance
                target = price + stop_distance * self.s.take_profit_rr
            else:
                stop = price + stop_distance
                target = price - stop_distance * self.s.take_profit_rr

            models = sorted({v.model for v in verdicts.values()})
            rationale = self._rationale(verdicts, gauge)
            evidence = {
                "verdicts": {name: v.model_dump(mode="json") for name, v in verdicts.items()},
                "sentiment_gauge": gauge,
                "atr": atr,
            }
            if context:
                evidence["context"] = context
            if fusion_context:
                evidence["fusion"] = {
                    "direction_score": fusion_context.fusion.direction_score,
                    "raw_confidence": fusion_context.fusion.raw_confidence,
                    "contributions": fusion_context.fusion.contributions,
                    "setup_quality": fusion_context.setup_quality.model_dump(),
                    "conflict": fusion_context.conflict.model_dump(),
                    "calibrated_confidence": fusion_context.calibrated_confidence,
                }
            mark("final", "pass", "proposal approved by risk engine")
            return SignalProposal(
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                confidence=round(confidence, 4),
                direction_score=(
                    round(fusion_context.fusion.direction_score, 4) if fusion_context else 0.0
                ),
                raw_confidence=round(confidence, 4),
                setup_quality=(
                    fusion_context.setup_quality.model_dump() if fusion_context else None
                ),
                conflicts=fusion_context.conflict.model_dump() if fusion_context else None,
                calibrated_confidence=(
                    fusion_context.calibrated_confidence if fusion_context else None
                ),
                entry=round(price, 8),
                stop=round(stop, 8),
                target=round(target, 8),
                size=round(size, 8),
                risk_amount=round(risk_amount, 2),
                expected_rr=self.s.take_profit_rr,
                rationale=rationale,
                evidence=evidence,
                model="+".join(models) if models else "unknown",
            )

    def _room_gate(
        self,
        symbol: str,
        side: Side,
        price: float,
        atr: float,
        context: dict | None,
        fusion: FusionContext | None,
    ) -> Rejection | None:
        """Room-to-target gate (V-MONSTER §29): INSUFFICIENT_ROOM.

        Only fires when the liquidity map found an opposing level and
        the after-cost room is below room_min_rr. No data never blocks.
        """
        if not self.s.room_gate_enabled:
            return None
        room = compute_room(
            side,
            price,
            atr,
            (context or {}).get("liquidity") or {},
            spread_pct=fusion.spread_pct if fusion else None,
            slippage_pct=self.s.slippage * 100.0,
            stop_mult=self.s.atr_stop_mult,
            min_room_rr=self.s.room_min_rr,
        )
        if room["insufficient"]:
            return Rejection(
                symbol=symbol,
                reason=f"insufficient room to target: {room['reason']}",
                no_trade_reason=NoTradeReason.INSUFFICIENT_ROOM.value,
            )
        return None

    def _no_trade_rejection(
        self, symbol: str, fusion: FusionContext | None
    ) -> Rejection | None:
        """Fusion-layer no-trade gates (spec §20).

        Returns a classified Rejection or None. Order is deliberate:
        conflicts first (a strong deterministic contradiction is the
        most serious refusal), then setup quality, then the opt-in
        HIGH_VOLATILITY / BAD_SPREAD / STATISTICAL_EDGE_UNKNOWN gates.
        """
        if not fusion:
            return None
        conflict = fusion.conflict
        if (
            conflict.state == ConflictState.CONFLICTED
            and self.s.conflict_block_conflicted
        ):
            axes = ", ".join(sorted({c.axis for c in conflict.conflicts}))
            details = "; ".join(c.detail for c in conflict.conflicts)
            reason = conflict.dominant_reason
            return Rejection(
                symbol=symbol,
                reason=f"deterministic conflicts on {len(conflict.conflicts)} axis(es) "
                f"({axes}): {details}",
                no_trade_reason=reason.value if reason else None,
            )
        if fusion.setup_quality.score < self.s.setup_quality_min:
            return Rejection(
                symbol=symbol,
                reason=(
                    f"setup quality {fusion.setup_quality.score:.2f} below minimum "
                    f"{self.s.setup_quality_min} ({fusion.setup_quality.detail})"
                ),
                no_trade_reason=NoTradeReason.LOW_SETUP_QUALITY.value,
            )
        if self.s.no_trade_high_volatility and fusion.regime == "high_volatility":
            return Rejection(
                symbol=symbol,
                reason="deterministic regime is high_volatility",
                no_trade_reason=NoTradeReason.HIGH_VOLATILITY.value,
            )
        if (
            self.s.no_trade_max_spread_pct > 0
            and fusion.spread_pct is not None
            and fusion.spread_pct > self.s.no_trade_max_spread_pct
        ):
            return Rejection(
                symbol=symbol,
                reason=(
                    f"spread {fusion.spread_pct:.2f}% above maximum "
                    f"{self.s.no_trade_max_spread_pct}%"
                ),
                no_trade_reason=NoTradeReason.BAD_SPREAD.value,
            )
        if self.s.require_statistical_edge and fusion.calibrated_confidence is None:
            return Rejection(
                symbol=symbol,
                reason="statistical edge unknown: calibration data insufficient",
                no_trade_reason=NoTradeReason.STATISTICAL_EDGE_UNKNOWN.value,
            )
        return None

    def _statistical_quality(
        self,
        session: Session,
        symbol: str,
        side: Side,
        fusion_context: FusionContext | None,
        gauge: dict | None,
        now: datetime | None,
    ) -> StatisticalQuality:
        """§31/§32 gate input: quality of signals similar to the candidate.

        Only resolved signals BEFORE this cycle are compared (the
        no-look-ahead boundary); MFE/MAE come from the positions those
        signals produced. Imported lazily: the analytics package pulls
        in the backtest engine, which imports the orchestrator — a cycle
        if imported at module level.
        """
        from trading_agent.analytics.stats import (
            candidate_features,
            resolved_signals,
            statistical_quality,
        )

        features = candidate_features(
            side.value,
            fusion_context.regime if fusion_context else None,
            gauge,
        )
        rows = resolved_signals(
            session,
            limit=self.s.calibration_window,
            before=now,
            symbol=symbol,
        )
        positions = {p.proposal_id: p for p in session.scalars(select(Position))}
        return statistical_quality(
            rows,
            features,
            min_sample=self.s.min_sample_for_statistics,
            positions=positions,
        )

    @staticmethod
    def _rationale(verdicts: dict[str, AgentVerdict], gauge: dict | None) -> str:
        parts = []
        for name, verdict in verdicts.items():
            notes = verdict.payload.get("reasoning") or verdict.payload.get("notes", "")
            parts.append(f"[{name}] {notes}")
        if gauge:
            label = gauge.get("source", "sentiment gauge")
            parts.append(f"[sentiment context] {label}: {gauge['value']} ({gauge['classification']})")
        return "\n".join(parts)
