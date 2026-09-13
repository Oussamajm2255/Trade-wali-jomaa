"""Risk engine — the deterministic layer with final authority.

The LLM proposes; this module disposes. Every gate is hard-coded,
unit-tested and audit-logged. Kill-switch state is persisted, so a halt
survives restarts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_agent.config import Settings
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
                )

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
            return SignalProposal(
                symbol=symbol,
                timeframe=timeframe,
                side=side,
                confidence=round(confidence, 4),
                entry=round(price, 8),
                stop=round(stop, 8),
                target=round(target, 8),
                size=round(size, 8),
                risk_amount=round(risk_amount, 2),
                expected_rr=self.s.take_profit_rr,
                rationale=rationale,
                evidence={
                    "verdicts": {name: v.model_dump() for name, v in verdicts.items()},
                    "sentiment_gauge": gauge,
                    "atr": atr,
                },
                model="+".join(models) if models else "unknown",
            )

    @staticmethod
    def _rationale(verdicts: dict[str, AgentVerdict], gauge: dict | None) -> str:
        parts = []
        for name, verdict in verdicts.items():
            notes = verdict.payload.get("notes", "")
            parts.append(f"[{name}] {notes}")
        if gauge:
            label = gauge.get("source", "sentiment gauge")
            parts.append(f"[sentiment context] {label}: {gauge['value']} ({gauge['classification']})")
        return "\n".join(parts)
