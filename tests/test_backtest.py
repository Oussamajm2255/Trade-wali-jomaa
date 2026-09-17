"""Backtest engine (spec §24/§25): candle-by-candle replay.

The engine's guarantees under test:
- No look-ahead: a signal made when candle N closes is filled at the
  OPEN of candle N+1 — never before, never at the same candle.
- Deterministic AI mode: every agent answers with its labelled
  heuristic fallback, so two replays over the same frames are
  bit-identical.
- Isolated state: the replay runs against its own database and risk
  state; the risk engine stays the final authority (kill-switch parity:
  analysis stops while halted, position management continues).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_agent.backtest.engine import BacktestEngine, BacktestError
from trading_agent.config import Settings
from trading_agent.store.db import init_engine, session_scope
from trading_agent.store.models import Position, Proposal, RiskState, SignalRecord

WARMUP = 250


def _settings(**overrides) -> Settings:
    base = dict(
        timeframe="15m",
        ohlcv_limit=WARMUP,
        snapshot_timeframes=[],
        htf_bias_filter_enabled=False,
        dxy_filter_enabled=False,
        session_filter_enabled=False,
        data_quality_min_candles=5,
        data_quality_allow_degraded=True,
        setup_quality_min=0.0,
        conflict_block_conflicted=False,
        min_confidence=0.35,
        deepseek_api_key=None,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    base.update(overrides)
    return Settings(**base)


def _gold_frame(n: int = 700, drift: float = 0.05, noise: float = 0.6) -> pd.DataFrame:
    """A smooth 15m uptrend with pullbacks: drift + wave + random-walk noise.

    The deterministic sine wave creates retracements away from the
    session highs, so the room-to-target gate (V-MONSTER §29) sees
    realistic distances on pullback entries and only rejects chases at
    the highs. A purely monotonic series would leave every liquidity
    level within a stop's distance and the gate would (correctly)
    refuse every entry.
    """
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    base = 3000.0 * (1.0 + drift * np.linspace(0.0, 1.0, n))
    wave = 15.0 * np.sin(2 * np.pi * np.arange(n) / 48.0)
    close = base + wave + np.cumsum(rng.normal(0.0, noise, n))
    open_ = np.empty(n)
    open_[0] = close[0] - 2.0
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 3.0
    low = np.minimum(open_, close) - 3.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )


def _dxy_frame(days: int = 18) -> pd.DataFrame:
    """A steadily declining 1h DXY frame (weak dollar = bullish gold).

    Starts well before the replay window so the historical gauge (7-day
    DXY return) is computable from the very first replayed candle.
    """
    n = days * 24
    idx = pd.date_range("2025-12-18", periods=n, freq="1h", tz="UTC")
    close = np.linspace(106.0, 96.0, n)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )


def _engine(gold: pd.DataFrame, dxy: pd.DataFrame | None = None, **kwargs):
    settings = kwargs.pop("settings", _settings())
    db_url = kwargs.pop("db_url", "sqlite:///:memory:")
    return BacktestEngine(
        settings,
        {"15m": gold},
        symbol="XAUUSD",
        timeframe="15m",
        dxy_frames=dxy,
        db_url=db_url,
        warmup=WARMUP,
        **kwargs,
    )


def _signal_ts(gold: pd.DataFrame, signal_id: str) -> pd.Timestamp:
    """Replay timestamp of a signal; SQLite returns naive datetimes."""
    with session_scope() as session:
        row = session.get(SignalRecord, signal_id)
    assert row is not None
    return pd.Timestamp(row.ts).tz_localize("UTC")


# --- engine run ----------------------------------------------------------


def test_run_produces_trades_and_propagates_outcomes() -> None:
    gold = _gold_frame()
    report = _engine(gold, _dxy_frame()).run()

    assert report.candles == len(gold) - WARMUP
    assert report.stats["trades"] >= 1
    assert all(t.signal_id for t in report.trades)

    # Isolated equity: start + sum of realized PnL, nothing else moves it.
    start = report.equity_curve[0][1]
    assert start == pytest.approx(10_000.0)
    final = report.equity_curve[-1][1]
    assert final == pytest.approx(start + sum(t.pnl for t in report.trades), abs=1e-3)
    assert report.stats["final_equity"] == pytest.approx(final)

    # Outcome propagation (spec §23): every closed trade's signal record
    # carries the same outcome and R multiple.
    with session_scope() as session:
        for trade in report.trades:
            row = session.get(SignalRecord, trade.signal_id)
            assert row is not None
            assert row.outcome == trade.outcome
            if trade.outcome in ("WIN", "LOSS"):
                assert row.r_multiple == pytest.approx(trade.r_multiple)


def test_fills_occur_at_next_candle_open_after_signal() -> None:
    gold = _gold_frame()
    report = _engine(gold, _dxy_frame(), spread_pct=0.0).run()
    trade = report.trades[0]
    ts = _signal_ts(gold, trade.signal_id)
    pos = gold.index.get_loc(ts)
    # The signal was made when candle `pos` closed; the fill is the open
    # of candle pos+1 — the first price actually available next.
    assert pos + 1 < len(gold)
    assert trade.entry == pytest.approx(float(gold.iloc[pos + 1]["open"]), abs=1e-6)


def test_replay_is_deterministic() -> None:
    gold, dxy = _gold_frame(), _dxy_frame()
    first = _engine(gold, dxy).run()
    second = _engine(gold, dxy).run()
    assert first.to_dict() == second.to_dict()


def test_spread_is_charged_on_entry() -> None:
    gold = _gold_frame()
    report = _engine(gold, _dxy_frame(), spread_pct=0.002).run()
    trade = next(t for t in report.trades if t.side == "long")
    ts = _signal_ts(gold, trade.signal_id)
    pos = gold.index.get_loc(ts)
    expected = float(gold.iloc[pos + 1]["open"]) * (1 + 0.002 / 2)
    assert trade.entry == pytest.approx(expected, abs=1e-6)


def test_room_gate_rejects_chasing_entries_in_backtest() -> None:
    """A monotonic series leaves every liquidity level within a stop's
    distance, so the room-to-target gate (V-MONSTER §29) rejects every
    entry as a chase — the reason `_gold_frame()` adds pullbacks so the
    gate has realistic distances to evaluate instead."""
    rng = np.random.default_rng(7)
    n = 700
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    base = 3000.0 * (1.0 + 0.05 * np.linspace(0.0, 1.0, n))
    close = base + np.cumsum(rng.normal(0.0, 0.6, n))
    open_ = np.empty(n)
    open_[0] = close[0] - 2.0
    open_[1:] = close[:-1]
    gold = pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + 3.0,
         "low": np.minimum(open_, close) - 3.0, "close": close, "volume": 100.0},
        index=idx,
    )

    gated = _engine(gold, _dxy_frame()).run()
    assert gated.stats["trades"] == 0

    settings = _settings()
    settings.room_gate_enabled = False
    ungated = _engine(gold, _dxy_frame(), settings=settings).run()
    assert ungated.stats["trades"] >= 1


# --- kill-switch parity --------------------------------------------------


def test_kill_switch_stops_analysis_but_manages_positions(tmp_path) -> None:
    """A pre-seeded open position is still managed after a drawdown halt.

    Paper-mode parity: the kill-switch stops NEW analysis, never the
    management of already-open positions — the crash candle closes the
    position, the realised loss breaches the drawdown limit, and the
    halt event is recorded on the exact candle.
    """
    db = f"sqlite:///{tmp_path / 'backtest.db'}"
    init_engine(db)
    crash_at = 300
    n = 450
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    open_ = np.full(n, 3000.0)
    close = 3000.0 + 0.01 * np.sin(np.arange(n))  # tiny noise, no NaN indicators
    high = np.maximum(open_, close) + 3.0
    low = np.minimum(open_, close) - 3.0
    low[crash_at] = 2950.0  # below the pre-seeded stop of 2990
    gold = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=idx,
    )

    with session_scope() as session:
        session.add(
            Proposal(
                id="kt-proposal",
                symbol="XAUUSD",
                timeframe="15m",
                side="long",
                confidence=0.7,
                entry=3000.0,
                stop=2990.0,
                target=3050.0,
                size=10.0,
                risk_amount=100.0,
                expected_rr=2.0,
                rationale="kill-switch parity",
                evidence={},
                model="test",
                status="approved",
            )
        )
        session.add(
            Position(
                proposal_id="kt-proposal",
                symbol="XAUUSD",
                side="long",
                size=10.0,
                entry=3000.0,
                stop=2990.0,
                target=3050.0,
                entry_fee=0.0,
                status="open",
            )
        )
        # init_engine already seeded RiskState(1) at 10k; the day field
        # is irrelevant to the drawdown logic under test.
        state = session.get(RiskState, 1)
        assert state is not None

    settings = _settings(max_drawdown=0.01, min_confidence=1.01)
    report = _engine(gold, None, settings=settings, db_url=db).run()

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.outcome == "LOSS"
    assert trade.entry == pytest.approx(3000.0)
    assert trade.exit_price == pytest.approx(2990.0)  # conservative stop-first fill
    assert report.stats["halt_events"] == 1
    assert "max drawdown breached" in report.halt_events[0]["reason"]
    assert report.halt_events[0]["ts"] == str(gold.index[crash_at])
    final = report.equity_curve[-1][1]
    assert final == pytest.approx(10_000.0 + trade.pnl, abs=1e-3)
    assert final < 10_000.0


def test_too_short_history_raises() -> None:
    gold = _gold_frame(n=200)
    with pytest.raises(BacktestError):
        _engine(gold, _dxy_frame()).run()
