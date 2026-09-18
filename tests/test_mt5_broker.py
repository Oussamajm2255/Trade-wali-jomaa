"""MT5 live-execution broker: unit tests against a fake MetaTrader5 module.

The real MetaTrader5 package needs a locally installed terminal, so the
module is injected into sys.modules and the broker's lazy import picks it
up. Pure logic under test: lot conversion, SL/TP payloads, requote retry,
reconciliation from deal history, kill-switch close-all, equity sync.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import trading_agent.store.db as db
from trading_agent.config import Settings
from trading_agent.execution.mt5 import MT5Broker, MT5Error
from trading_agent.execution.paper import PaperBroker
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Side, SignalProposal
from trading_agent.store import actions
from trading_agent.store.models import Position

RETCODE_DONE = 10009
RETCODE_REQUOTE = 10004
RETCODE_REJECT = 10013


def _build_fake_mt5(state: dict) -> types.ModuleType:
    fake = types.ModuleType("MetaTrader5")
    fake.TRADE_ACTION_DEAL = 1
    fake.ORDER_TYPE_BUY = 0
    fake.ORDER_TYPE_SELL = 1
    fake.TRADE_RETCODE_DONE = RETCODE_DONE
    fake.TRADE_RETCODE_REQUOTE = RETCODE_REQUOTE
    fake.ORDER_TIME_GTC = 0
    fake.ORDER_FILLING_IOC = 1
    fake.TIMEFRAME_M1 = 1
    fake.TIMEFRAME_M5 = 5
    fake.TIMEFRAME_M15 = 15
    fake.TIMEFRAME_M30 = 30
    fake.TIMEFRAME_H1 = 60
    fake.TIMEFRAME_H4 = 240
    fake.TIMEFRAME_D1 = 1440

    def initialize(**kwargs):
        state["initialized"] = True
        return True

    def login(login_id, password=None, server=None):
        state["login"] = (login_id, server)
        return True

    def last_error():
        return (-1, "fake error")

    def account_info():
        return SimpleNamespace(login=12345, equity=state["equity"], balance=10_000.0)

    def symbol_info(symbol):
        if symbol == "XAUUSD":
            return SimpleNamespace(
                name="XAUUSD", trade_contract_size=100.0, volume_min=0.01,
                volume_max=200.0, volume_step=0.01, digits=2, point=0.01,
                trade_stops_level=0, visible=True,
            )
        return None

    def symbol_info_tick(symbol):
        return SimpleNamespace(ask=4350.0, bid=4349.5)

    def symbols_get(*args, **kwargs):
        return [SimpleNamespace(name="XAUUSD")]

    def symbol_select(symbol, enable=True):
        return True

    def order_send(request):
        state["orders"].append(dict(request))
        retcode = state["retcodes"].pop(0) if state["retcodes"] else RETCODE_DONE
        if retcode == RETCODE_DONE:
            ticket = state["next_ticket"]
            state["next_ticket"] += 1
            state["positions"].append(
                SimpleNamespace(
                    ticket=ticket, symbol=request["symbol"], comment=request.get("comment", ""),
                    type=request["type"], volume=request["volume"], price_open=request["price"],
                    sl=request.get("sl", 0.0), tp=request.get("tp", 0.0),
                    magic=request.get("magic", 0),
                )
            )
            return SimpleNamespace(retcode=RETCODE_DONE, comment="done",
                                   price=request["price"], deal=ticket, order=ticket,
                                   commission=0.0)
        return SimpleNamespace(retcode=retcode, comment="rejected",
                               price=0.0, deal=0, order=0, commission=0.0)

    def positions_get(**kwargs):
        out = list(state["positions"])
        if "symbol" in kwargs:
            out = [p for p in out if p.symbol == kwargs["symbol"]]
        if "magic" in kwargs:
            out = [p for p in out if getattr(p, "magic", None) == kwargs["magic"]]
        if "ticket" in kwargs:
            out = [p for p in out if p.ticket == kwargs["ticket"]]
        return out

    def history_deals_get(**kwargs):
        return state["deals"].get(kwargs.get("position"), [])

    def copy_rates_from_pos(symbol, timeframe, start_pos, count):
        state.setdefault("rates", []).append((symbol, timeframe, count))
        return [
            {"time": 1720000000 + i * 900, "open": 1.0, "high": 1.1, "low": 0.9,
             "close": 1.05, "tick_volume": 100, "spread": 1, "real_volume": 0}
            for i in range(count)
        ]

    def shutdown():
        state["shutdown"] = True
        return True

    fake.initialize = initialize
    fake.login = login
    fake.last_error = last_error
    fake.account_info = account_info
    fake.symbol_info = symbol_info
    fake.symbol_info_tick = symbol_info_tick
    fake.symbols_get = symbols_get
    fake.symbol_select = symbol_select
    fake.order_send = order_send
    fake.positions_get = positions_get
    fake.history_deals_get = history_deals_get
    fake.copy_rates_from_pos = copy_rates_from_pos
    fake.shutdown = shutdown
    return fake


@pytest.fixture
def fake_mt5(monkeypatch) -> dict:
    state = {
        "initialized": False,
        "shutdown": False,
        "equity": 10_000.0,
        "orders": [],
        "positions": [],
        "deals": {},
        "retcodes": [],
        "next_ticket": 100,
        "rates": [],
    }
    monkeypatch.setitem(sys.modules, "MetaTrader5", _build_fake_mt5(state))
    return state


def make_proposal(**overrides) -> SignalProposal:
    defaults = dict(
        symbol="XAUUSD", timeframe="1h", side=Side.LONG, confidence=0.7,
        entry=4350.0, stop=4300.0, target=4450.0, size=12.34,
        risk_amount=50.0, expected_rr=2.0, rationale="test",
        evidence={}, model="test",
    )
    defaults.update(overrides)
    return SignalProposal(**defaults)


def approve_proposal(proposal: SignalProposal) -> SignalProposal:
    pid = actions.save_proposal(proposal)
    approved = actions.decide_proposal(pid, approve=True)
    assert approved is not None
    return approved


# ------------------------------------------------------------------ sizing


def test_fetch_ohlcv_reads_broker_candles(fake_mt5, base_settings):
    """Data-bridge contract: broker candles in a UTC-indexed OHLCV frame."""
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    df = broker.fetch_ohlcv("XAUUSD", "15m", 300)
    assert len(df) == 300
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None  # UTC epoch -> aware index
    assert fake_mt5["rates"][0] == ("XAUUSD", 15, 300)


def test_fetch_ohlcv_unsupported_timeframe_raises(fake_mt5, base_settings):
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    with pytest.raises(MT5Error, match="unsupported timeframe"):
        broker.fetch_ohlcv("XAUUSD", "7m", 10)


def test_lots_conversion_from_units(fake_mt5, base_settings):
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    broker.connect()
    symbol_info = broker._import_mt5().symbol_info("XAUUSD")
    assert broker._lots_from_size(symbol_info, 12.34) == 0.12  # 12.34 oz / 100 per lot
    assert broker._lots_from_size(symbol_info, 100.0) == 1.0


def test_lots_below_min_raises(fake_mt5, base_settings):
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    broker.connect()
    symbol_info = broker._import_mt5().symbol_info("XAUUSD")
    with pytest.raises(MT5Error, match="below minimum"):
        broker._lots_from_size(symbol_info, 0.5)


# -------------------------------------------------------------- open orders


def test_open_position_payload_with_sl_tp(fake_mt5, base_settings, seeded):
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    position = broker.open_position(approve_proposal(make_proposal()))
    assert position is not None and position.status == "open"
    assert position.broker_ticket == 100
    req = fake_mt5["orders"][0]
    assert req["type"] == 0  # BUY
    assert req["volume"] == 0.12
    assert req["sl"] == 4300.0 and req["tp"] == 4450.0
    assert req["magic"] == 770313
    assert req["comment"].startswith("ta:")
    # never a naked position: SL/TP must be on every order
    assert req["sl"] and req["tp"]


def test_open_position_requote_then_done(fake_mt5, base_settings, seeded):
    fake_mt5["retcodes"] = [RETCODE_REQUOTE]
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    position = broker.open_position(approve_proposal(make_proposal()))
    assert position is not None
    assert len(fake_mt5["orders"]) == 2


def test_open_position_failure_raises_no_position(fake_mt5, base_settings, seeded):
    fake_mt5["retcodes"] = [RETCODE_REJECT]
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    with pytest.raises(MT5Error, match="order failed"):
        broker.open_position(approve_proposal(make_proposal()))
    with db.SessionLocal() as session:
        assert session.query(Position).count() == 0


# ------------------------------------------------------------------ close-all


def test_close_all_flattens_broker_positions(fake_mt5, base_settings, seeded):
    fake_mt5["positions"] = [
        SimpleNamespace(ticket=1, symbol="XAUUSD", type=0, volume=0.1, magic=770313),
        SimpleNamespace(ticket=2, symbol="XAUUSD", type=1, volume=0.2, magic=770313),
    ]
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    assert broker.close_all() == 2
    assert fake_mt5["orders"][0]["type"] == 1  # SELL closes the BUY
    assert fake_mt5["orders"][1]["type"] == 0  # BUY closes the SELL


# -------------------------------------------------------------- reconcile


def test_manage_reconciles_closed_from_history(fake_mt5, base_settings, seeded):
    with db.SessionLocal() as session:
        session.add(Position(
            proposal_id="p", symbol="XAUUSD", side="long", size=10.0,
            entry=4350.0, stop=4300.0, target=4450.0, broker_ticket=99, status="open",
        ))
        session.commit()
    fake_mt5["deals"][99] = [
        SimpleNamespace(entry=0, price=4350.0, profit=0.0),
        SimpleNamespace(entry=1, price=4330.0, profit=-50.0),
    ]
    broker = MT5Broker(base_settings, RiskEngine(base_settings))
    events = broker.manage("XAUUSD")
    assert len(events) == 1
    assert events[0]["exit_reason"] == "broker_close"
    assert events[0]["pnl"] == -50.0
    with db.SessionLocal() as session:
        row = session.query(Position).one()
        assert row.status == "closed" and row.pnl == -50.0
    equity = actions.get_risk_state().equity
    assert equity == pytest.approx(9950.0)


# -------------------------------------------------------------- equity sync


def test_set_equity_engages_kill_switch(base_settings, seeded):
    risk = RiskEngine(base_settings)
    risk.set_equity(8_500.0)  # -15% on the day
    halted, reason = risk.is_halted()
    assert halted and "daily loss" in reason


def test_set_equity_ratchets_peak_up_only(base_settings, seeded):
    risk = RiskEngine(base_settings)
    risk.set_equity(10_500.0)
    risk.set_equity(10_200.0)  # a drawdown below peak but within limits
    state = actions.get_risk_state()
    assert state.peak_equity == pytest.approx(10_500.0)
    assert not state.halted


# -------------------------------------------------------------- paper parity


def test_paper_close_all_applies_slippage(base_settings, seeded):
    broker = PaperBroker(base_settings, RiskEngine(base_settings))
    assert broker.open_position(approve_proposal(make_proposal())) is not None
    events = broker.close_all({"XAUUSD": 4350.0})
    assert len(events) == 1
    assert events[0]["exit_reason"] == "manual_close"
    assert events[0]["exit_price"] == pytest.approx(4350.0 * (1 - 0.0005))


# -------------------------------------------------------------- proposal revert


def test_revert_proposal_back_to_pending(base_settings, seeded):
    proposal = approve_proposal(make_proposal())
    actions.revert_proposal(proposal.id, "execution failed: test")
    assert actions.get_proposal(proposal.id).status == "pending"
