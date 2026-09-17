"""Signal stability + final real-time revalidation (V-MONSTER §32/§56).

The guarantees under test:
- Stability classification: STABLE/FRAGILE/VERY_FRAGILE from ±1 tick /
  −1 candle perturbations — score sensitivity (tick), entry/stop
  sensitivity (candle, in ATRs), speed-state flip on the shortened
  frame; NEUTRAL is not classified; insufficient history is FRAGILE,
  never fabricated.
- The fusion context carries the stability verdict with every signal
  record (alongside the Phase D trigger).
- The orchestrator's final revalidation: after every gate passed, one
  fresh tick must still support the proposal — price drift, spread and
  data age abort the send with a recorded rejection (no Telegram);
  no tick source, no data or a fetch error fails open.
"""
from __future__ import annotations

import types

import pandas as pd

from trading_agent.agents.orchestrator import Orchestrator
from trading_agent.config import Settings
from trading_agent.data.gold import GoldData
from trading_agent.data.market import MarketData
from trading_agent.data.snapshot import build_market_snapshot
from trading_agent.execution.mt5 import MT5Broker
from trading_agent.fusion import stability as stability_module
from trading_agent.fusion.engine import build_fusion_context
from trading_agent.fusion.stability import StabilityState, compute_stability
from trading_agent.fusion.types import FusionResult, NoTradeReason
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, Side, SignalProposal
from trading_agent.store.db import session_scope
from trading_agent.store.models import SignalRecord

from sqlalchemy import select

from test_snapshot import FakeMarket, fresh_gauge, make_df


def make_df_quiet(n: int = 100) -> pd.DataFrame:
    """Uniform frame: TR = 1.0, ATR ~= 1.0, closes rising 0.01."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("15min"), periods=n, freq="15min", tz="UTC")
    close = 2000.0 + pd.Series(range(n), dtype=float).to_numpy() * 0.01
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 100.0},
        index=idx,
    )


class Snap:
    def __init__(self, price: float = 2000.0, candles: dict | None = None,
                 speed: dict | None = None, timeframe: str = "15m") -> None:
        self.price = price
        self.candles = candles or {}
        self.speed = speed or {}
        self.entry_timeframe = timeframe


def _constant_quality(score: float = 0.8):
    return lambda side, snapshot, settings: types.SimpleNamespace(score=score)


# ------------------------------------------------------------- stability


def test_neutral_side_not_classified() -> None:
    out = compute_stability(Side.NEUTRAL, Snap(), Settings())
    assert out["state"] is None
    assert out["components"] == {}


def test_insufficient_history_is_fragile() -> None:
    out = compute_stability(Side.LONG, Snap(candles={"15m": make_df_quiet(2)}), Settings())
    assert out["state"] == StabilityState.FRAGILE
    assert "insufficient history" in out["detail"]


def test_quiet_frame_is_stable(monkeypatch) -> None:
    monkeypatch.setattr(stability_module, "compute_setup_quality", _constant_quality())
    snap = Snap(price=2000.99, candles={"15m": make_df_quiet()})
    out = compute_stability(Side.LONG, snap, Settings())
    assert out["state"] == StabilityState.STABLE
    # TR is 1.0 everywhere: ATR ~= 1.0, close move 0.01 -> tiny axes.
    assert out["components"]["entry_atr"] < 0.05
    assert out["components"]["stop_atr"] < 0.05
    assert out["components"]["speed_flip"] is False


def test_score_sensitivity_marks_fragile(monkeypatch) -> None:
    def price_sensitive(side, snapshot, settings):
        score = 0.8 if abs(snapshot.price - 2000.99) < 1e-9 else 0.7
        return types.SimpleNamespace(score=score)

    monkeypatch.setattr(stability_module, "compute_setup_quality", price_sensitive)
    snap = Snap(price=2000.99, candles={"15m": make_df_quiet()})
    out = compute_stability(Side.LONG, snap, Settings())
    assert out["state"] == StabilityState.FRAGILE
    assert out["components"]["score_tick"] == 0.1


def test_score_sensitivity_marks_very_fragile(monkeypatch) -> None:
    def price_sensitive(side, snapshot, settings):
        score = 0.8 if abs(snapshot.price - 2000.99) < 1e-9 else 0.5
        return types.SimpleNamespace(score=score)

    monkeypatch.setattr(stability_module, "compute_setup_quality", price_sensitive)
    snap = Snap(price=2000.99, candles={"15m": make_df_quiet()})
    out = compute_stability(Side.LONG, snap, Settings())
    assert out["state"] == StabilityState.VERY_FRAGILE
    assert out["components"]["score_tick"] == 0.3


def test_big_last_candle_moves_stop_to_very_fragile(monkeypatch) -> None:
    monkeypatch.setattr(stability_module, "compute_setup_quality", _constant_quality())
    candles = make_df_quiet()
    last = candles.index[-1]
    candles.loc[last, "high"] = float(candles.loc[last, "close"]) + 5.0
    candles.loc[last, "low"] = float(candles.loc[last, "close"]) - 5.0
    snap = Snap(price=2000.99, candles={"15m": candles})
    out = compute_stability(Side.LONG, snap, Settings())
    assert out["state"] == StabilityState.VERY_FRAGILE
    assert out["components"]["stop_atr"] > 0.25


def test_speed_flip_marks_fragile(monkeypatch) -> None:
    monkeypatch.setattr(stability_module, "compute_setup_quality", _constant_quality())
    monkeypatch.setattr(
        stability_module, "compute_market_speed", lambda *a, **kw: {"state": "EXTREME"}
    )
    snap = Snap(price=2000.99, candles={"15m": make_df_quiet()}, speed={"state": "NORMAL"})
    out = compute_stability(Side.LONG, snap, Settings())
    assert out["state"] == StabilityState.FRAGILE
    assert out["components"]["speed_flip"] is True


def test_fusion_context_carries_stability() -> None:
    settings = Settings(
        htf_timeframe="4h",
        snapshot_timeframes=["1h", "4h", "1d"],
        telegram_bot_token="",
        telegram_chat_id="",
    )
    snap = build_market_snapshot(FakeMarket(gauge=fresh_gauge()), "XAUUSD", settings, "15m")
    verdicts = {}
    ctx = build_fusion_context(snap, verdicts, settings, "4h")
    assert ctx.stability["state"] in (None, "STABLE", "FRAGILE", "VERY_FRAGILE")
    assert "components" in ctx.stability
    assert ctx.trigger["quality"] is not None  # Phase D rides alongside


# --------------------------------------------------------- revalidation


class TickMarket(FakeMarket):
    def __init__(self, tick_result=None, **kw) -> None:
        super().__init__(**kw)
        self._tick_result = tick_result
        self.tick_calls = 0

    def tick(self, symbol: str) -> dict | None:
        self.tick_calls += 1
        if self._tick_result is not None:
            return dict(self._tick_result)
        df = self.fetch_ohlcv(symbol, "15m", 300)
        return {"price": float(df["close"].iloc[-1]), "spread": None, "age_s": 1.0}


def _settings(**overrides) -> Settings:
    base = dict(
        htf_timeframe="4h",
        snapshot_timeframes=["1h"],
        min_confidence=0.0,
        setup_quality_min=0.0,
        conflict_block_conflicted=False,
        htf_bias_filter_enabled=False,
        room_gate_enabled=False,
        deepseek_api_key=None,
        telegram_bot_token="",
        telegram_chat_id="",
    )
    base.update(overrides)
    return Settings(**base)


def _orch(settings: Settings, market) -> Orchestrator:
    return Orchestrator(settings, market, RiskEngine(settings))


def _force_long(orch: Orchestrator, monkeypatch) -> None:
    monkeypatch.setattr(
        orch,
        "_fuse",
        lambda verdicts: FusionResult(side=Side.LONG, direction_score=0.8, raw_confidence=0.8),
    )


def _last_close() -> float:
    return float(make_df(300, "15m")["close"].iloc[-1])


def test_revalidation_pass_when_tick_matches(monkeypatch) -> None:
    settings = _settings()
    orch = _orch(settings, TickMarket(gauge=fresh_gauge()))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        gates = [g for g in (row.gates or []) if g["gate"] == "revalidation"]
        assert gates and gates[0]["status"] == "pass"
        # Phase F stability rides the stored fusion dict (NEUTRAL fuses
        # are stored honestly as unclassified).
        assert "state" in row.fusion["stability"]
        assert row.fusion["trigger"]["quality"] is not None


def test_revalidation_aborts_on_drift(monkeypatch) -> None:
    settings = _settings()
    market = TickMarket(gauge=fresh_gauge(), tick_result={"price": 1.0})
    orch = _orch(settings, market)
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.SIGNAL_INVALIDATED.value
    assert "drifted" in result.reason
    assert market.tick_calls == 1
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        assert row.final_decision == "rejected"
        assert row.no_trade_reason == NoTradeReason.SIGNAL_INVALIDATED.value
        gates = [g for g in (row.gates or []) if g["gate"] == "revalidation"]
        assert gates and gates[0]["status"] == "reject"


def test_revalidation_aborts_on_spread(monkeypatch) -> None:
    entry = _last_close()
    settings = _settings(no_trade_max_spread_pct=0.05)
    market = TickMarket(gauge=fresh_gauge(), tick_result={"price": entry, "spread": 3.0})
    orch = _orch(settings, market)
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.BAD_SPREAD.value
    assert "spread" in result.reason


def test_revalidation_aborts_on_stale_tick(monkeypatch) -> None:
    entry = _last_close()
    settings = _settings(revalidate_max_age_s=60)
    market = TickMarket(gauge=fresh_gauge(), tick_result={"price": entry, "age_s": 9999})
    orch = _orch(settings, market)
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.SIGNAL_INVALIDATED.value
    assert "age" in result.reason


def test_revalidation_disabled_never_checks(monkeypatch) -> None:
    settings = _settings(final_revalidation_enabled=False)
    market = TickMarket(gauge=fresh_gauge(), tick_result={"price": 1.0})
    orch = _orch(settings, market)
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    assert market.tick_calls == 0
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        assert not [g for g in (row.gates or []) if g["gate"] == "revalidation"]


def test_revalidation_without_tick_source_fails_open(monkeypatch) -> None:
    settings = _settings()
    orch = _orch(settings, FakeMarket(gauge=fresh_gauge()))
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)
    with session_scope() as session:
        row = session.scalars(select(SignalRecord)).first()
        gates = [g for g in (row.gates or []) if g["gate"] == "revalidation"]
        assert gates == [{"gate": "revalidation", "status": "pass", "detail": "no tick source"}]


def test_revalidation_tick_failure_fails_open(monkeypatch) -> None:
    market = TickMarket(gauge=fresh_gauge())
    monkeypatch.setattr(market, "tick", lambda symbol: (_ for _ in ()).throw(RuntimeError("down")))
    settings = _settings()
    orch = _orch(settings, market)
    _force_long(orch, monkeypatch)
    result, *_ = orch.run_full("XAUUSD", "15m")
    assert isinstance(result, SignalProposal)


# ------------------------------------------------------------ tick sources


def test_mt5_tick_none_when_disconnected() -> None:
    broker = MT5Broker(Settings(), RiskEngine(Settings()))
    assert broker.tick("XAUUSD") is None


def test_market_data_tick_reads_ticker(monkeypatch) -> None:
    market = MarketData("binance")
    fake_client = types.SimpleNamespace(
        fetch_ticker=lambda symbol: {
            "last": 2000.0,
            "bid": 1999.5,
            "ask": 2000.5,
            "timestamp": None,
        }
    )
    monkeypatch.setattr(market, "_client", lambda: fake_client)
    out = market.tick("XAUUSD")
    assert out == {"price": 2000.0, "spread": 1.0, "age_s": None}


def test_market_data_tick_failure_is_none(monkeypatch) -> None:
    market = MarketData("binance")
    monkeypatch.setattr(
        market, "_client", lambda: (_ for _ in ()).throw(RuntimeError("down"))
    )
    assert market.tick("XAUUSD") is None


def test_gold_tick_delegates_to_mt5() -> None:
    class _Mt5:
        def tick(self, symbol):
            return {"price": 2000.0, "spread": 0.5, "age_s": 0.1}

    market = GoldData(mt5=_Mt5())
    assert market.tick("XAUUSD") == {"price": 2000.0, "spread": 0.5, "age_s": 0.1}


def test_gold_tick_none_without_sources() -> None:
    market = GoldData()  # no MT5, last_source "unknown" -> honest None
    assert market.tick("XAUUSD") is None
