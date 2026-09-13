"""Paper broker: simulated fills, fees, slippage and exit management.

Fill model: latest closed candle + slippage + taker fee.
Exit model: conservative — when a candle's range spans both stop and
target, the stop is assumed to hit first.
"""
from __future__ import annotations

import logging

from trading_agent.config import Settings
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Side, SignalProposal, utcnow
from trading_agent.store.db import session_scope
from trading_agent.store.models import AuditLog, Position, Proposal

logger = logging.getLogger(__name__)


class PaperBroker:
    def __init__(self, settings: Settings, risk: RiskEngine) -> None:
        self.s = settings
        self.risk = risk

    def open_position(self, proposal: SignalProposal) -> Position | None:
        """Open a paper position from an approved proposal."""
        if proposal.id is None:
            raise ValueError("proposal must be persisted before opening a position")
        with session_scope() as session:
            row = session.get(Proposal, proposal.id)
            if row is None or row.status != "approved":
                return None
            if row.position is not None:
                return row.position  # idempotent: never double-open
            slippage = self.s.slippage
            entry = (
                proposal.entry * (1 + slippage)
                if proposal.side == Side.LONG
                else proposal.entry * (1 - slippage)
            )
            fee = proposal.size * entry * self.s.fee_rate
            position = Position(
                proposal_id=proposal.id,
                symbol=proposal.symbol,
                side=proposal.side.value,
                size=proposal.size,
                entry=round(entry, 8),
                stop=proposal.stop,
                target=proposal.target,
                entry_fee=round(fee, 8),
                status="open",
            )
            session.add(position)
            session.flush()
            session.add(
                AuditLog(
                    level="INFO",
                    event="position_opened",
                    detail={
                        "proposal_id": proposal.id,
                        "symbol": proposal.symbol,
                        "side": proposal.side.value,
                        "size": proposal.size,
                        "entry": round(entry, 8),
                        "stop": proposal.stop,
                        "target": proposal.target,
                    },
                )
            )
            return position

    def manage(self, symbol: str, candle) -> list[dict]:
        """Check open positions against a candle; close on stop/target hits."""
        events: list[dict] = []
        with session_scope() as session:
            positions = session.query(Position).filter(
                Position.status == "open", Position.symbol == symbol
            ).all()
            for pos in positions:
                exit_price, reason = self._exit_check(pos, candle)
                if exit_price is None:
                    continue
                events.append(self._close(session, pos, exit_price, reason))
        return events

    def close_all(self, prices: dict[str, float]) -> list[dict]:
        """Manual flatten: close every open paper position at market.

        `prices` maps symbol -> last close; symbols without a price are
        skipped (never guess a fill).
        """
        events: list[dict] = []
        with session_scope() as session:
            positions = session.query(Position).filter(Position.status == "open").all()
            for pos in positions:
                price = prices.get(pos.symbol)
                if price is None:
                    logger.warning("no price for %s; skipping manual close", pos.symbol)
                    continue
                direction = 1.0 if pos.side == Side.LONG.value else -1.0
                exit_price = price * (1 - self.s.slippage * direction)
                events.append(self._close(session, pos, exit_price, "manual_close"))
        return events

    def _close(self, session, pos: Position, exit_price: float, reason: str) -> dict:
        """Close one position at the given price; realise PnL. Caller owns
        the session/transaction."""
        exit_fee = pos.size * exit_price * self.s.fee_rate
        direction = 1.0 if pos.side == Side.LONG.value else -1.0
        pnl = round(
            (exit_price - pos.entry) * pos.size * direction
            - pos.entry_fee
            - exit_fee,
            8,
        )
        pos.status = "closed"
        pos.exit_price = round(exit_price, 8)
        pos.exit_fee = round(exit_fee, 8)
        pos.pnl = pnl
        pos.closed_at = utcnow()
        pos.exit_reason = reason
        self.risk.realize_pnl(session, pnl)
        session.add(
            AuditLog(
                level="INFO",
                event="position_closed",
                detail={
                    "position_id": pos.id,
                    "symbol": pos.symbol,
                    "exit_reason": reason,
                    "exit_price": round(exit_price, 8),
                    "pnl": pnl,
                },
            )
        )
        return {
            "position_id": pos.id,
            "symbol": pos.symbol,
            "exit_reason": reason,
            "exit_price": round(exit_price, 8),
            "pnl": pnl,
        }

    @staticmethod
    def _exit_check(pos: Position, candle) -> tuple[float | None, str | None]:
        high = float(candle["high"])
        low = float(candle["low"])
        if pos.side == Side.LONG.value:
            if low <= pos.stop:
                # Conservative: assume the stop filled first at the stop price.
                return min(float(candle["open"]), pos.stop), "stop_loss"
            if high >= pos.target:
                return pos.target, "take_profit"
        else:
            if high >= pos.stop:
                return max(float(candle["open"]), pos.stop), "stop_loss"
            if low <= pos.target:
                return pos.target, "take_profit"
        return None, None
