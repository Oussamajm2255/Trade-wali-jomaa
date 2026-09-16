"""Paper broker: simulated fills, fees, slippage and exit management.

Fill model: latest closed candle + slippage + taker fee.
Exit model: conservative — when a candle's range spans both stop and
target, the stop is assumed to hit first.

Phase 5 (spec §23): while a position is open the broker tracks its path
candle-by-candle (MFE/MAE, bars held) and on close the outcome engine
classifies the trade and fills the signal record + agent tracks.
"""
from __future__ import annotations

import logging

from trading_agent.config import Settings
from trading_agent.outcome.engine import finalize_position
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

    def manage(self, symbol: str, candle, now=None) -> list[dict]:
        """Check open positions against a candle; close on stop/target hits.

        Before the exit check the candle's range is folded into each
        position's MFE/MAE path (spec §23), so outcome statistics are
        complete on close. `now` (historical replays) stamps closed_at
        and the risk state's daily rollover with the candle's own time.
        """
        events: list[dict] = []
        with session_scope() as session:
            positions = session.query(Position).filter(
                Position.status == "open", Position.symbol == symbol
            ).all()
            for pos in positions:
                self._update_path(pos, candle)
                exit_price, reason = self._exit_check(pos, candle)
                if exit_price is None:
                    continue
                events.append(self._close(session, pos, exit_price, reason, now))
        return events

    def close_all(self, prices: dict[str, float], now=None) -> list[dict]:
        """Manual flatten: close every open paper position at market.

        `prices` maps symbol -> last close; symbols without a price are
        skipped (never guess a fill). `now` (historical replays) stamps
        closed_at with the liquidation timestamp instead of wall clock.
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
                events.append(self._close(session, pos, exit_price, "manual_close", now))
        return events

    def _close(self, session, pos: Position, exit_price: float, reason: str, now=None) -> dict:
        """Close one position at the given price; realise PnL. Caller owns
        the session/transaction. The outcome engine (spec §23) classifies
        the trade and fills the signal record + agent tracks."""
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
        pos.closed_at = now or utcnow()
        pos.exit_reason = reason
        finalize_position(session, pos)
        self.risk.realize_pnl(session, pnl, now)
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
                    "outcome": pos.outcome,
                    "r_multiple": pos.r_multiple,
                },
            )
        )
        return {
            "position_id": pos.id,
            "symbol": pos.symbol,
            "exit_reason": reason,
            "exit_price": round(exit_price, 8),
            "pnl": pnl,
            "outcome": pos.outcome,
            "r_multiple": pos.r_multiple,
        }

    @staticmethod
    def _update_path(pos: Position, candle) -> None:
        """Fold one candle's range into the position's MFE/MAE path."""
        pos.bars_open = (pos.bars_open or 0) + 1
        high = float(candle["high"])
        low = float(candle["low"])
        if pos.side == Side.LONG.value:
            pos.mfe_price = max(pos.mfe_price if pos.mfe_price is not None else high, high)
            pos.mae_price = min(pos.mae_price if pos.mae_price is not None else low, low)
        else:
            pos.mfe_price = max(pos.mfe_price if pos.mfe_price is not None else low, low)
            pos.mae_price = min(pos.mae_price if pos.mae_price is not None else high, high)

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
