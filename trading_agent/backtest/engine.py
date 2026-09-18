"""Historical backtesting (spec §24): candle-by-candle replay.

No look-ahead, ever:
- The ONLY look-ahead boundary is the HistoricalMarket: every frame is
  cut at `<= now` before it reaches the snapshot builder, so no
  downstream module can see a candle that had not closed when the
  signal was made.
- Staleness / session checks are anchored to the replay `now`
  (threaded through quality/snapshot/risk/paper brokers).
- A signal produced when candle N closes is filled at the OPEN of
  candle N+1 — the first price actually available next.
- Opt-in realistic execution (§69/§70): the fill is delayed by the
  same human reaction window Phase G uses and priced with its drift
  projection, so replays model the trader's latency instead of the
  idealised next-open fill. The model never peeks forward: the
  projected price comes from the signal's own stored timing payload
  (pace at signal time), not from future candles.
- Historical DXY gauges are recomputed from daily DXY closes known at
  the replay timestamp, with the same formula the live gauge uses.

Deterministic AI mode (spec §25): the backtest always runs with a copy
of the settings where deepseek_api_key is None, so llm_enabled=False
and every agent answers with its labelled heuristic fallback —
reproducible, offline, no agent-failure blocks. The risk engine remains
the same final authority it is in paper/live mode; its kill-switch,
daily-loss and drawdown limits are enforced on the backtest's isolated
equity state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.data.gold import GoldData
from trading_agent.data.market import MarketDataError
from trading_agent.execution.paper import PaperBroker
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, Side, SignalProposal
from trading_agent.store import actions
from trading_agent.store.db import init_engine, session_scope
from trading_agent.store.models import AuditLog, Position, Proposal, SignalRecord

logger = logging.getLogger(__name__)


class BacktestError(RuntimeError):
    """Raised when a backtest cannot be configured or run."""


def _parse_ts(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class HistoricalMarket:
    """Market facade over pre-fetched frames, sliced at the replay time."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame],
        dxy_frames: pd.DataFrame | None = None,
        gauge_source: str = "DXY dollar index (historical)",
    ) -> None:
        self.frames = {tf: df for tf, df in frames.items() if df is not None and len(df)}
        self.dxy = dxy_frames
        self.gauge_source = gauge_source
        self._now: pd.Timestamp | None = None
        self.last_source = "historical"

    def set_now(self, now) -> None:
        self._now = pd.Timestamp(now)
        if self._now.tzinfo is None:
            self._now = self._now.tz_localize("UTC")

    def _cut(self, df: pd.DataFrame, limit: int) -> pd.DataFrame:
        if self._now is None:
            return df.tail(limit).copy()
        return df[df.index <= self._now].tail(limit).copy()

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=300, ttl=None) -> pd.DataFrame:
        df = self.frames.get(timeframe)
        if df is None or df.empty:
            raise MarketDataError(f"no historical frame for {timeframe}")
        return self._cut(df, limit)

    def dxy_ohlcv(self, timeframe="15m", limit=300) -> pd.DataFrame:
        if self.dxy is None or self.dxy.empty:
            raise MarketDataError("no historical DXY data")
        return self._cut(self.dxy, limit)

    def sentiment_gauge(self) -> dict | None:
        """DXY gauge recomputed from daily closes known at the replay time."""
        if self.dxy is None or self.dxy.empty:
            return None
        now = self._now or self.dxy.index[-1]
        daily = self.dxy[self.dxy.index <= now][["close"]]
        gauge = GoldData._compute_gauge(daily, self.gauge_source)
        if gauge is not None:
            gauge["ts"] = str(now)  # fresh at replay time by construction
        return gauge


class BacktestBroker(PaperBroker):
    """Paper broker with spread costs and next-candle-open fills.

    The engine supplies the fill price (the next candle's open) so the
    broker never peeks forward itself; entry and exit each pay half the
    round-trip spread. Everything else — path tracking, conservative
    stop-first exits, outcome finalisation — is the paper broker's.
    """

    def __init__(self, settings: Settings, risk: RiskEngine, spread_pct: float = 0.0) -> None:
        super().__init__(settings, risk)
        self.spread_pct = spread_pct

    def _apply_spread(self, price: float, direction: float) -> float:
        return price * (1 + direction * self.spread_pct / 2.0)

    def open_position_at(
        self,
        proposal: SignalProposal,
        market_price: float,
        opened_at=None,
        *,
        fill_model: str = "next_open",
        slippage_pct: float | None = None,
    ) -> Position | None:
        """Fill an approved proposal at the engine-supplied price.

        The §24 model supplies the next candle's open; the realistic
        model supplies the Phase G drift projection and charges the
        configured slippage on top (spread is charged either way).
        """
        if proposal.id is None:
            raise ValueError("proposal must be persisted before opening a position")
        direction = 1.0 if proposal.side == Side.LONG else -1.0
        entry = self._apply_spread(float(market_price), direction)
        if slippage_pct:
            entry = entry * (1 + direction * slippage_pct)
        with session_scope() as session:
            row = session.get(Proposal, proposal.id)
            if row is None or row.status != "approved":
                return None
            if row.position is not None:
                return row.position  # idempotent
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
                opened_at=opened_at,
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
                        "fill_model": fill_model,
                    },
                )
            )
            return position

    def manage(self, symbol: str, candle, now=None) -> list[dict]:
        """Paper-broker exit management; exits pay half the spread."""
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
                direction = 1.0 if pos.side == Side.LONG.value else -1.0
                exit_price = self._apply_spread(exit_price, -direction)
                events.append(self._close(session, pos, exit_price, reason, now))
        return events


@dataclass
class _PendingFill:
    """One auto-approved proposal waiting for its fill window."""

    proposal: SignalProposal
    fill_ts: pd.Timestamp | None  # None = next candle open (§24 model)
    drift_price: float | None  # Phase G projected entry (realistic model)


def reaction_fill(
    proposal: SignalProposal,
    signal_ts: pd.Timestamp,
    timing: dict | None,
    settings: Settings,
) -> _PendingFill:
    """Realistic fill scheduling: Phase G's reaction window + drift.

    The delay is the signal's stored timing `reaction_s` when available
    (what Phase G assumed at send time); otherwise the configured
    budget (`user_reaction_seconds` + `telegram_latency_s`). The fill
    price is the Phase G projection: entry plus the pace-based drift
    over the reaction window (`pace_per_minute` from the same payload),
    with the broker's spread and slippage charged on top. Without a
    timing payload the fill is reaction-delayed only (drift 0.0) —
    never fabricated, and never peeks at future candles.
    """
    payload = timing or {}
    if isinstance(payload.get("reaction_s"), (int, float)):
        reaction_s = float(payload["reaction_s"])
    else:
        reaction_s = float(
            getattr(settings, "user_reaction_seconds", 120.0) or 120.0
        ) + float(getattr(settings, "telegram_latency_s", 3.0) or 3.0)
    pace = payload.get("pace_per_minute")
    drift_px = (
        float(pace) * reaction_s / 60.0 if isinstance(pace, (int, float)) else 0.0
    )
    sign = 1.0 if proposal.side == Side.LONG else -1.0
    return _PendingFill(
        proposal=proposal,
        fill_ts=pd.Timestamp(signal_ts) + pd.Timedelta(seconds=reaction_s),
        drift_price=float(proposal.entry) + sign * drift_px,
    )


def _signal_timing(proposal_id: str) -> dict | None:
    """The Phase G timing payload stored with the signal, if any.

    Looked up through the proposal link: the decision row carries the
    proposal id, not the signal id, and the record is linked back to it
    by `link_signal_proposal` before the approval happens.
    """
    try:
        with session_scope() as session:
            record = session.scalar(
                select(SignalRecord).where(SignalRecord.proposal_id == proposal_id)
            )
        if record is not None:
            return (record.fusion or {}).get("timing") or None
    except Exception:  # noqa: BLE001 - fill scheduling never kills the replay
        return None
    return None


@dataclass
class BacktestTrade:
    """One closed trade of the replay."""

    signal_id: str | None
    opened_at: datetime | None
    closed_at: datetime | None
    side: str
    entry: float | None
    exit_price: float | None
    pnl: float | None
    outcome: str | None
    r_multiple: float | None
    bars_open: int
    exit_reason: str | None

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "opened_at": str(self.opened_at) if self.opened_at else None,
            "closed_at": str(self.closed_at) if self.closed_at else None,
            "side": self.side,
            "entry": self.entry,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "outcome": self.outcome,
            "r_multiple": self.r_multiple,
            "bars_open": self.bars_open,
            "exit_reason": self.exit_reason,
        }


@dataclass
class BacktestReport:
    """Everything the replay produced: trades, equity curve, halt events."""

    symbol: str
    timeframe: str
    start_ts: str
    end_ts: str
    candles: int
    db_url: str
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    halt_events: list[dict] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        wins = [t for t in self.trades if t.outcome == "WIN"]
        losses = [t for t in self.trades if t.outcome == "LOSS"]
        resolved = len(wins) + len(losses)
        gross_win = sum(t.pnl for t in wins if t.pnl is not None)
        gross_loss = sum(t.pnl for t in losses if t.pnl is not None)
        rs = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        peak, max_dd = 0.0, 0.0
        for _, equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "breakeven": sum(1 for t in self.trades if t.outcome == "BREAKEVEN"),
            "expired": sum(1 for t in self.trades if t.outcome == "EXPIRED"),
            "invalidated": sum(1 for t in self.trades if t.outcome == "INVALIDATED"),
            "win_rate": round(len(wins) / resolved, 4) if resolved else None,
            "profit_factor": round(gross_win / abs(gross_loss), 4) if gross_loss < 0 else None,
            "expectancy_r": round(sum(rs) / len(rs), 4) if rs else None,
            "total_r": round(sum(rs), 4) if rs else 0.0,
            "total_pnl": round(sum(t.pnl for t in self.trades if t.pnl is not None), 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "final_equity": round(self.equity_curve[-1][1], 2) if self.equity_curve else None,
            "halt_events": len(self.halt_events),
        }

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "candles": self.candles,
            "db_url": self.db_url,
            "stats": self.stats,
            "trades": [t.to_dict() for t in self.trades],
            "equity_curve": [{"ts": ts, "equity": eq} for ts, eq in self.equity_curve],
            "halt_events": self.halt_events,
        }


class BacktestEngine:
    """Candle-by-candle historical replay of the full pipeline (spec §24).

    Each closed candle: pending fills at its open, position management
    over its range, then analysis at its close. The analysis result
    (proposal or rejection) is stored as a signal record in the isolated
    backtest database; approved proposals are auto-executed at the next
    candle's open (the human-in-the-loop is modelled as "approve
    everything the risk engine approves").

    With `backtest_realistic_execution` the fill is instead scheduled at
    signal time + the human reaction window (the signal's stored Phase G
    timing) and priced with the Phase G drift projection plus the
    configured slippage; the position is managed from the first candle
    that opens after the fill. Either way the fill never depends on
    prices the model has not reached yet.
    """

    def __init__(
        self,
        settings: Settings,
        frames: dict[str, pd.DataFrame],
        symbol: str = "XAUUSD",
        timeframe: str | None = None,
        dxy_frames: pd.DataFrame | None = None,
        start: str | None = None,
        end: str | None = None,
        db_url: str | None = None,
        spread_pct: float | None = None,
        warmup: int | None = None,
    ) -> None:
        self.settings = settings
        self.symbol = symbol
        self.timeframe = timeframe or settings.timeframe
        self.db_url = db_url or settings.backtest_db_url
        self.spread_pct = settings.backtest_spread_pct if spread_pct is None else spread_pct
        self.warmup = warmup if warmup is not None else settings.ohlcv_limit
        self.frames = {tf: df for tf, df in frames.items() if df is not None and len(df)}
        self.dxy_frames = dxy_frames
        self.start = _parse_ts(start) if start else None
        self.end = _parse_ts(end) if end else None

    def run(self) -> BacktestReport:
        tf = self.timeframe
        entry = self.frames.get(tf)
        if entry is None or entry.empty:
            raise BacktestError(f"no historical frame for {self.symbol} {tf}")
        entry = entry[~entry.index.duplicated(keep="last")].sort_index()
        if self.start is not None:
            entry = entry[entry.index >= self.start]
        if self.end is not None:
            entry = entry[entry.index <= self.end]
        if len(entry) <= self.warmup:
            raise BacktestError(
                f"need more than warmup={self.warmup} candles in range, got {len(entry)}"
            )

        # Deterministic AI mode (spec §25): no API key -> llm_enabled
        # False -> every agent answers with its labelled heuristic
        # fallback. Reproducible and offline; the risk engine stays the
        # same final authority it is in paper/live mode.
        settings = self.settings.model_copy(
            update={"deepseek_api_key": None, "timeframe": tf}
        )
        realistic = bool(getattr(settings, "backtest_realistic_execution", False))
        init_engine(self.db_url)
        market = HistoricalMarket(self.frames, self.dxy_frames)
        risk = RiskEngine(settings)
        broker = BacktestBroker(settings, risk, self.spread_pct)
        orchestrator = Orchestrator(settings, market, risk)

        trades: list[BacktestTrade] = []
        halt_events: list[dict] = []
        state0 = actions.get_risk_state()
        start_equity = state0.equity if state0 else settings.paper_starting_equity
        equity_curve: list[tuple[str, float]] = [
            (str(entry.index[self.warmup - 1]), start_equity)
        ]
        pending: list[_PendingFill] = []

        for i in range(self.warmup, len(entry)):
            ts = entry.index[i]
            now = ts.to_pydatetime()
            candle = entry.iloc[i]
            market.set_now(ts)

            # 1. Fills whose reaction window has elapsed (realistic
            # model) or plain next-open fills (§24 model) — at THIS
            # candle's open (the first price actually available).
            still_pending: list[_PendingFill] = []
            for item in pending:
                if item.fill_ts is not None and item.fill_ts > ts:
                    still_pending.append(item)  # reaction window not over yet
                    continue
                if item.fill_ts is not None:
                    fill_price = (
                        item.drift_price
                        if item.drift_price is not None
                        else float(item.proposal.entry)
                    )
                    opened_at = item.fill_ts.to_pydatetime()
                    fill_model = "realistic_reaction_drift"
                    slippage_pct = settings.slippage
                else:
                    fill_price = float(candle["open"])
                    opened_at = now
                    fill_model = "next_open"
                    slippage_pct = None
                position = broker.open_position_at(
                    item.proposal,
                    fill_price,
                    opened_at,
                    fill_model=fill_model,
                    slippage_pct=slippage_pct,
                )
                if position is not None:
                    logger.info(
                        "backtest fill %s @ %.8g", item.proposal.signal_id, position.entry
                    )
            pending = still_pending

            halted_before, _ = risk.is_halted()

            # 2. Manage open positions against this candle's range.
            for event in broker.manage(self.symbol, candle, now):
                trades.append(self._trade_from_event(event))

            halted_now, halt_reason = risk.is_halted()
            if halted_now and not halted_before:
                halt_events.append({"ts": str(ts), "reason": halt_reason})
                logger.warning("backtest kill-switch engaged at %s: %s", ts, halt_reason)

            state = actions.get_risk_state()
            equity_curve.append((str(ts), state.equity if state else 0.0))

            # 3. Analysis at this candle's close — skipped while halted
            # (paper-mode parity) and on the last candle (its fill would
            # need a next open that does not exist).
            if halted_now or i == len(entry) - 1:
                continue
            result, _verdicts, _entry, _gauge = orchestrator.run_full(
                self.symbol, tf, now=now
            )
            if isinstance(result, Rejection):
                continue
            proposal_id = actions.save_proposal(result)
            actions.link_signal_proposal(result.signal_id, proposal_id)
            approved = actions.decide_proposal(
                proposal_id, approve=True, note="backtest auto-approval"
            )
            if approved is not None:
                if not realistic:
                    pending.append(_PendingFill(approved, None, None))
                else:
                    timing = _signal_timing(proposal_id)
                    pending.append(reaction_fill(approved, ts, timing, settings))

        # Liquidate whatever is still open at the last close, stamped with
        # the replay timestamp (deterministic; never wall clock).
        last_close = float(entry.iloc[-1]["close"])
        for event in broker.close_all({self.symbol: last_close}, now=entry.index[-1].to_pydatetime()):
            trades.append(self._trade_from_event(event))
        state = actions.get_risk_state()
        if state is not None:
            equity_curve.append((str(entry.index[-1]), state.equity))

        return BacktestReport(
            symbol=self.symbol,
            timeframe=tf,
            start_ts=str(entry.index[self.warmup]),
            end_ts=str(entry.index[-1]),
            candles=len(entry) - self.warmup,
            db_url=self.db_url,
            trades=trades,
            equity_curve=equity_curve,
            halt_events=halt_events,
        )

    @staticmethod
    def _trade_from_event(event: dict) -> BacktestTrade:
        with session_scope() as session:
            pos = session.get(Position, event["position_id"])
            if pos is not None:
                signal = session.scalar(
                    select(SignalRecord).where(SignalRecord.proposal_id == pos.proposal_id)
                )
                return BacktestTrade(
                    signal_id=signal.signal_id if signal else None,
                    opened_at=pos.opened_at,
                    closed_at=pos.closed_at,
                    side=pos.side,
                    entry=pos.entry,
                    exit_price=event.get("exit_price"),
                    pnl=event.get("pnl"),
                    outcome=event.get("outcome"),
                    r_multiple=event.get("r_multiple"),
                    bars_open=pos.bars_open or 0,
                    exit_reason=event.get("exit_reason"),
                )
        return BacktestTrade(
            signal_id=None,
            opened_at=None,
            closed_at=None,
            side="",
            entry=None,
            exit_price=event.get("exit_price"),
            pnl=event.get("pnl"),
            outcome=event.get("outcome"),
            r_multiple=event.get("r_multiple"),
            bars_open=0,
            exit_reason=event.get("exit_reason"),
        )
