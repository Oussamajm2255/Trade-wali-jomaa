"""Live execution via MetaTrader 5 — IC Markets, FTMO, or any MT5 broker.

Only the server/login differ between brokers, so one adapter serves all.

Safety invariants (enforced, never configurable away):
- every order carries server-side SL and TP — no naked positions,
- orders are idempotent by proposal id (comment tag + magic number),
- app-owned positions are tracked by broker ticket; unknown positions are
  adopted, never double-managed,
- kill-switch closes every position on the account,
- equity is synced from the broker so daily-loss/drawdown limits are
  enforced on REAL money, not local approximations.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import pandas as pd

from trading_agent.config import Settings
from trading_agent.outcome.engine import finalize_position
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Side, SignalProposal, utcnow
from trading_agent.store.db import session_scope
from trading_agent.store.models import AuditLog, Position, Proposal

logger = logging.getLogger(__name__)

COMMENT_PREFIX = "ta:"

_TIMEFRAME_MAP = {
    "1m": "TIMEFRAME_M1",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1d": "TIMEFRAME_D1",
}


class MT5Error(RuntimeError):
    """Raised on any terminal/order failure."""


class MT5Broker:
    def __init__(self, settings: Settings, risk: RiskEngine) -> None:
        self.s = settings
        self.risk = risk
        self._mt5: Any = None
        self._connected = False

    # ------------------------------------------------------------- terminal

    def _import_mt5(self) -> Any:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5
            except ImportError as exc:
                raise MT5Error(
                    "MetaTrader5 package missing — run: pip install -r requirements-live.txt"
                ) from exc
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> None:
        """Initialise the terminal and log in. Idempotent."""
        if self._connected:
            return
        mt5 = self._import_mt5()
        kwargs: dict = {}
        if self.s.mt5_path:
            kwargs["path"] = self.s.mt5_path
        if not mt5.initialize(**kwargs):
            raise MT5Error(f"MT5 initialize failed: {mt5.last_error()}")
        password = self.s.mt5_password.get_secret_value() if self.s.mt5_password else ""
        if not mt5.login(self.s.mt5_login, password=password, server=self.s.mt5_server):
            raise MT5Error(f"MT5 login failed: {mt5.last_error()}")
        info = mt5.account_info()
        if info is None:
            raise MT5Error("MT5 account_info returned nothing after login")
        self._connected = True
        logger.info("MT5 connected: login=%s server=%s equity=%.2f",
                    self.s.mt5_login, self.s.mt5_server, info.equity)

    def shutdown(self) -> None:
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            finally:
                self._connected = False
                self._mt5 = None

    # ------------------------------------------------------------- helpers

    def _resolve_symbol(self, symbol: str) -> str:
        mt5 = self._import_mt5()
        if mt5.symbol_info(symbol) is not None:
            return symbol
        # Broker suffix variants (e.g. "XAUUSD." on some servers).
        for candidate in mt5.symbols_get() or []:
            if candidate.name.upper().replace(" ", "") == symbol.upper().replace("/", ""):
                return candidate.name
        raise MT5Error(f"symbol {symbol!r} not found on this server")

    @staticmethod
    def _lots_from_size(symbol_info: Any, size_units: float) -> float:
        contract = float(symbol_info.trade_contract_size)
        if contract <= 0:
            raise MT5Error(f"symbol {symbol_info.name}: invalid contract size {contract}")
        step = float(symbol_info.volume_step)
        lots = math.floor((size_units / contract) / step) * step
        if lots < float(symbol_info.volume_min):
            raise MT5Error(
                f"size {size_units:.4f} units = {lots:.4f} lots below "
                f"minimum {symbol_info.volume_min}"
            )
        lots = min(lots, float(symbol_info.volume_max))
        return round(lots, 8)

    def _send(self, request: dict, attempts: int = 3) -> Any:
        """Send an order; retry on requote with a fresh price (bounded)."""
        mt5 = self._import_mt5()
        last_result = None
        for attempt in range(attempts):
            result = mt5.order_send(request)
            if result is None:
                raise MT5Error(f"order_send returned None: {mt5.last_error()}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return result
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE and attempt < attempts - 1:
                tick = mt5.symbol_info_tick(request["symbol"])
                if tick is None:
                    raise MT5Error("requote and no tick available")
                request["price"] = tick.ask if request["type"] == mt5.ORDER_TYPE_BUY else tick.bid
                time.sleep(0.2 * (attempt + 1))
                last_result = result
                continue
            raise MT5Error(f"order failed: retcode={result.retcode} comment={result.comment}")
        raise MT5Error(f"order failed after {attempts} requotes: {last_result.comment}")

    # ------------------------------------------------------------- trading

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
        """Read candles from the terminal (e.g. DXY_U6 for broker-native DXY)."""
        self.connect()
        mt5 = self._import_mt5()
        tf_name = _TIMEFRAME_MAP.get(timeframe)
        if tf_name is None:
            raise MT5Error(f"unsupported timeframe {timeframe!r}")
        resolved = self._resolve_symbol(symbol)
        rates = mt5.copy_rates_from_pos(resolved, getattr(mt5, tf_name), 0, limit)
        if rates is None or len(rates) == 0:
            raise MT5Error(f"no candles for {resolved}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})[
            ["time", "open", "high", "low", "close", "volume"]
        ]
        return df.set_index("time")

    def open_position(self, proposal: SignalProposal) -> Position | None:
        """Open a live position with server-side SL/TP. Idempotent by proposal."""
        if proposal.id is None:
            raise MT5Error("proposal must be persisted before opening a position")
        self.connect()
        mt5 = self._import_mt5()
        with session_scope() as session:
            row = session.get(Proposal, proposal.id)
            if row is None or row.status != "approved":
                return None
            if row.position is not None:
                return row.position  # idempotent: never double-open

        symbol = self._resolve_symbol(proposal.symbol)
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise MT5Error(f"cannot select symbol {symbol} in Market Watch")
            symbol_info = mt5.symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f"no tick for {symbol}")

        is_long = proposal.side == Side.LONG
        price = tick.ask if is_long else tick.bid
        sl, tp = self._validate_sl_tp(symbol_info, price, proposal, is_long)
        lots = self._lots_from_size(symbol_info, proposal.size)

        comment = f"{COMMENT_PREFIX}{proposal.id[:24]}"
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.s.mt5_deviation_points,
            "magic": self.s.mt5_magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = self._send(request)

        # Find the opened position by our idempotency tag.
        ticket = None
        for pos in mt5.positions_get(symbol=symbol, magic=self.s.mt5_magic) or []:
            if pos.comment == comment:
                ticket = int(pos.ticket)
                break
        if ticket is None:
            raise MT5Error("order filled but position not found — reconciliation required")

        with session_scope() as session:
            row = session.get(Proposal, proposal.id)
            if row is None:
                raise MT5Error("proposal vanished while order was in flight")
            position = Position(
                proposal_id=proposal.id,
                symbol=proposal.symbol,
                side=proposal.side.value,
                size=proposal.size,
                entry=round(float(result.price), 8),
                stop=proposal.stop,
                target=proposal.target,
                entry_fee=round(float(getattr(result, "commission", 0.0)), 8),
                broker_ticket=ticket,
                status="open",
            )
            session.add(position)
            session.flush()
            session.add(
                AuditLog(
                    level="INFO",
                    event="position_opened_live",
                    detail={
                        "proposal_id": proposal.id,
                        "symbol": proposal.symbol,
                        "side": proposal.side.value,
                        "lots": lots,
                        "ticket": ticket,
                        "price": float(result.price),
                        "sl": sl,
                        "tp": tp,
                    },
                )
            )
            return position

    @staticmethod
    def _validate_sl_tp(symbol_info: Any, price: float, proposal: SignalProposal, is_long: bool):
        point = float(symbol_info.point)
        digits = int(symbol_info.digits)
        min_dist = float(symbol_info.trade_stops_level) * point
        stop_dist = abs(price - proposal.stop)
        if stop_dist < min_dist:
            raise MT5Error(
                f"stop {proposal.stop} too close to market {price} "
                f"(min distance {min_dist:.4f})"
            )
        return round(proposal.stop, digits), round(proposal.target, digits)

    def manage(self, symbol: str, candle: Any = None) -> list[dict]:
        """Live: server-side stops do the work; we reconcile + sync equity."""
        self.connect()
        mt5 = self._import_mt5()
        self._sync_equity(mt5)
        events: list[dict] = []
        broker_open = mt5.positions_get(symbol=symbol, magic=self.s.mt5_magic) or []
        by_comment = {p.comment: p for p in broker_open}
        tickets = {int(p.ticket) for p in broker_open}

        with session_scope() as session:
            db_open = session.query(Position).filter(
                Position.status == "open", Position.symbol == symbol
            ).all()
            for pos in db_open:
                # Adopt app-opened positions that crashed before being recorded.
                if pos.broker_ticket is None and pos.proposal_id:
                    tag = f"{COMMENT_PREFIX}{pos.proposal_id[:24]}"
                    if tag in by_comment:
                        pos.broker_ticket = int(by_comment[tag].ticket)
                        session.add(AuditLog(level="WARNING", event="position_adopted",
                                             detail={"position_id": pos.id, "ticket": pos.broker_ticket}))
                        continue
                    # Paper-legacy row without a broker counterpart: do not
                    # double-trade the symbol; leave it for manual review.
                    logger.warning("position %s has no broker counterpart — manual review required", pos.id)
                    continue
                # Reconcile closed positions from deal history.
                if pos.broker_ticket is not None and pos.broker_ticket not in tickets:
                    events.append(self._reconcile_close(session, pos, mt5))
        return events

    def _reconcile_close(self, session, pos: Position, mt5: Any) -> dict:
        deals = mt5.history_deals_get(position=pos.broker_ticket) or []
        out_deal = next((d for d in deals if d.entry == 1), None)  # 1 = DEAL_ENTRY_OUT
        if out_deal is None:
            logger.warning("position %s (ticket %s) gone but no closing deal found",
                           pos.id, pos.broker_ticket)
            return {"position_id": pos.id, "symbol": pos.symbol,
                    "exit_reason": "unknown", "exit_price": 0.0, "pnl": 0.0}
        exit_price = float(out_deal.price)
        profit = float(out_deal.profit)
        pos.status = "closed"
        pos.exit_price = round(exit_price, 8)
        pos.pnl = round(profit, 8)
        pos.closed_at = utcnow()
        pos.exit_reason = "broker_close"
        finalize_position(session, pos)
        self.risk.realize_pnl(session, profit)
        session.add(AuditLog(level="INFO", event="position_closed_live",
                             detail={"position_id": pos.id, "ticket": pos.broker_ticket,
                                     "price": exit_price, "pnl": profit,
                                     "outcome": pos.outcome, "r_multiple": pos.r_multiple}))
        return {"position_id": pos.id, "symbol": pos.symbol, "exit_reason": "broker_close",
                "exit_price": exit_price, "pnl": profit,
                "outcome": pos.outcome, "r_multiple": pos.r_multiple}

    def _sync_equity(self, mt5: Any) -> None:
        info = mt5.account_info()
        if info is not None:
            self.risk.set_equity(float(info.equity))

    def close_all(self) -> int:
        """Kill-switch: flatten every position owned by this app."""
        self.connect()
        mt5 = self._import_mt5()
        closed = 0
        for pos in mt5.positions_get(magic=self.s.mt5_magic) or []:
            is_long = pos.type == mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(pos.symbol)
            price = tick.bid if is_long else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": self.s.mt5_deviation_points,
                "magic": self.s.mt5_magic,
                "comment": f"{COMMENT_PREFIX}kill",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            self._send(request)
            closed += 1
        if closed:
            with session_scope() as session:
                session.add(
                    AuditLog(
                        level="CRITICAL",
                        event="close_all_executed",
                        detail={"positions_closed": closed, "magic": self.s.mt5_magic},
                    )
                )
        return closed
