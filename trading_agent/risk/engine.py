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
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_agent.config import Settings
from trading_agent.fusion.types import ConflictState, FusionContext, NoTradeReason
from trading_agent.schema.types import AgentVerdict, Rejection, Side, SignalProposal
from trading_agent.store.db import session_scope
from trading_agent.store.models import AuditLog, Position, RiskState

logger = logging.getLogger(__name__)

MIN_NOTIONAL = 5.0  # refuse dust-sized paper positions (USD)


class RiskEngine:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    # ------------------------------------------------------------------ state

    def _get_or_create_state(self, session: Session) -> RiskState:
        state = session.get(RiskState, 1)
        if state is None:
            equity = self.s.paper_starting_equity
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        self._roll_day(state)
        return state

    @staticmethod
    def _roll_day(state: RiskState) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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

    def realize_pnl(self, session: Session, pnl: float) -> None:
        """Apply realised PnL to equity, update peak, enforce limits.

        Runs inside the caller's session/transaction — the caller owns
        commit/rollback (the paper broker calls this within session_scope).
        """
        state = self._get_or_create_state(session)
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
    ) -> SignalProposal | Rejection:
        """Apply every hard gate; return a proposal or a logged rejection."""
        with session_scope() as session:
            state = self._get_or_create_state(session)
            self._enforce_limits(session, state)

            if state.halted:
                return Rejection(symbol=symbol, reason=f"kill-switch engaged: {state.halt_reason}")
            if side == Side.NEUTRAL:
                return Rejection(symbol=symbol, reason="fused signal is neutral")
            if confidence < self.s.min_confidence:
                return Rejection(
                    symbol=symbol,
                    reason=f"confidence {confidence:.2f} below minimum {self.s.min_confidence}",
                    no_trade_reason=NoTradeReason.LOW_CONFIDENCE.value,
                )

            # --- No-trade gates (spec §20), fusion-layer inputs. ---
            no_trade = self._no_trade_rejection(symbol, fusion_context)
            if no_trade:
                return no_trade

            # DXY concurrency hard gate (gold): the dollar must agree with
            # the trade direction or the signal is refused — no exceptions.
            if self.s.dxy_filter_enabled and gauge and gauge.get("kind") == "dxy":
                value = float(gauge["value"])
                if side == Side.LONG and value < self.s.dxy_long_min:
                    return Rejection(
                        symbol=symbol,
                        reason=(
                            f"DXY concurrency: gauge {value:.0f} not weak-dollar "
                            f"(need >= {self.s.dxy_long_min:.0f}) for a LONG"
                        ),
                    )
                if side == Side.SHORT and value > self.s.dxy_short_max:
                    return Rejection(
                        symbol=symbol,
                        reason=(
                            f"DXY concurrency: gauge {value:.0f} not strong-dollar "
                            f"(need <= {self.s.dxy_short_max:.0f}) for a SHORT"
                        ),
                    )
            # HTF bias hard gate (multi-timeframe): the 4h trend must agree
            # with the entry direction; a choppy HTF blocks both sides.
            if self.s.htf_bias_filter_enabled and htf_bias:
                bias = htf_bias.get("bias")
                if side == Side.LONG and bias != "bull":
                    return Rejection(
                        symbol=symbol,
                        reason=(
                            f"HTF bias: {bias} on {self.s.htf_timeframe} blocks LONG "
                            f"({htf_bias.get('detail', '')})"
                        ),
                    )
                if side == Side.SHORT and bias != "bear":
                    return Rejection(
                        symbol=symbol,
                        reason=(
                            f"HTF bias: {bias} on {self.s.htf_timeframe} blocks SHORT "
                            f"({htf_bias.get('detail', '')})"
                        ),
                    )
            open_positions = list(session.scalars(select(Position).where(Position.status == "open")))
            if any(p.symbol == symbol for p in open_positions):
                return Rejection(symbol=symbol, reason="position already open for this symbol")
            if len(open_positions) >= self.s.max_positions:
                return Rejection(symbol=symbol, reason=f"max positions reached ({self.s.max_positions})")
            if atr <= 0 or price <= 0:
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
                "verdicts": {name: v.model_dump() for name, v in verdicts.items()},
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
