"""§38 Telegram upgrade: full proposal format from the signal record,
concise rejection alerts gated by telegram_rejection_alerts."""

import pytest

from trading_agent.config import Settings
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.schema.types import Side, SignalProposal

DXY_WEAK = {"value": 60, "classification": "Bullish (USD weak)", "kind": "dxy"}


@pytest.fixture
def telegram_settings() -> Settings:
    return Settings(telegram_bot_token="123:abc", telegram_chat_id="987")


def _proposal() -> SignalProposal:
    return SignalProposal(
        id="pid-1",
        signal_id="sig-1",
        symbol="XAUUSD",
        timeframe="15m",
        side=Side.LONG,
        confidence=0.72,
        entry=4350.0,
        stop=4300.0,
        target=4450.0,
        size=1.2,
        risk_amount=50.0,
        expected_rr=2.0,
        rationale="r",
        evidence={},
        model="test",
    )


def _record(**overrides) -> dict:
    """SignalRecord-shaped dict mirroring _record_signal output (§22)."""
    record = {
        "signal_id": "sig-1",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "market_snapshot": {
            "symbol": "XAUUSD",
            "timeframe": "15m",
            "data_quality": "good",
            "last_close": 4350.0,
            "atr_14": 12.5,
            "mtf_biases": {
                "1d": {"bias": "long", "detail": "x"},
                "4h": {"bias": "long", "detail": "y"},
                "1h": {"bias": "long", "detail": "z"},
                "15m": {"bias": "long", "detail": "w"},
            },
            "alignment": {"label": "aligned"},
            "regime": {"regime": "trend_up"},
            "structure": {"bos": True, "choch": True, "fvgs": [1], "sweeps": True},
            "dxy_context": {"classification": "Bullish (USD weak)", "trend": "bull"},
            "dxy_gauge": DXY_WEAK,
            "session_context": {"session": "LONDON"},
            "price": 4350.0,
        },
        "ai_outputs": {
            "technical": {
                "agent": "technical",
                "source": "llm",
                "payload": {"bias": "long", "conviction": 0.8, "reasoning": "r"},
            },
            "regime": {
                "agent": "regime",
                "source": "fallback",
                "payload": {"bias": "long", "conviction": 0.6},
            },
        },
        "fusion": {
            "direction_score": 0.7,
            "raw_confidence": 0.7,
            "calibrated_confidence": 0.65,
            "regime": "trend_up",
            "spread_pct": 0.05,
        },
        "setup_quality": {
            "score": 0.75,
            "components": {
                "mtf": 0.8,
                "structure": 0.7,
                "regime": 0.6,
                "dxy": 0.9,
                "volatility": 0.3,
                "session": 0.4,
                "risk_reward": 0.8,
            },
        },
        "conflicts": {"state": "ALIGNED", "conflicts": [], "conflict_score": 0.0},
        "gates": [
            {"gate": "ai_signal", "status": "pass"},
            {"gate": "setup_quality", "status": "pass"},
            {"gate": "risk", "status": "pass"},
        ],
        "final_decision": "proposal",
        "sl": 4300.0,
        "tp": 4450.0,
        "risk_amount": 50.0,
        "size": 1.2,
    }
    record.update(overrides)
    return record


def test_full_proposal_carries_every_section(telegram_settings):
    notifier = TelegramNotifier(telegram_settings)
    text = notifier.proposal_message(_proposal(), record=_record(), proposal_id="pid-9")
    assert "🎯 SIGNAL XAUUSD — LONG" in text
    assert "Entrée : 4,350.00" in text
    assert "Stop : 4,300.00" in text
    assert "Objectif : 4,450.00" in text
    assert "RR : 2.00 | Taille : 1.2000 | Risque : 50.00 $" in text
    assert "Confiance brute : 0.72 | Calibrée : 0.65" in text
    assert "Qualité du setup : 0.75 (conflits : ALIGNED)" in text
    assert "Biais MTF : 1D haussière · 4H haussière · 1H haussière · 15m haussière" in text
    assert (
        "Régime : tendance haussière | DXY : Bullish (USD weak)"
        " | Tendance DXY : haussière | Session : Londres" in text
    )
    assert "Structure : BOS, CHoCH, FVG, balayage de liquidité" in text
    assert "Soutient : " in text and "alignement MTF fort (0.80)" in text
    assert "Contredit : " in text and "volatilité faible (0.30)" in text
    assert "Invalidation : setup invalidé si le prix clôture au-delà du SL (4,300)" in text
    assert "Gates : ai_signal ✓ | setup_quality ✓ | risk ✓" in text
    assert "TECHNICAL : long (0.80)" in text
    assert "REGIME : long (0.60) [heuristique]" in text
    assert "Version : LEGACY_BASELINE" in text
    assert "⏳ EN ATTENTE D'APPROBATION HUMAINE" in text
    assert "approve pid-9" in text
    assert "reject pid-9" in text


def test_tick_volume_vwap_is_labelled_in_proposal(telegram_settings):
    """§7 honesty: broker tick-volume VWAP never reads as traded-volume VWAP."""
    record = _record()
    record["market_snapshot"]["vwap"] = {
        "available": True,
        "session_vwap": 4352.1,
        "daily_vwap": 4355.0,
        "state": "above",
        "volume_basis": "tick",
    }
    notifier = TelegramNotifier(telegram_settings)
    text = notifier.proposal_message(_proposal(), record=record, proposal_id="pid-10")
    assert "VWAP session : 4,352.10 (volume tick)" in text
    assert "VWAP jour : 4,355.00" in text
    # The real-volume VWAP never carries the tick marker.
    record["market_snapshot"]["vwap"]["volume_basis"] = "real"
    text = notifier.proposal_message(_proposal(), record=record, proposal_id="pid-11")
    assert "VWAP session : 4,352.10 (volume tick)" not in text
    assert "VWAP session : 4,352.10" in text


def test_proposal_missing_context_sections_are_omitted(telegram_settings):
    record = _record()
    record["market_snapshot"] = {"last_close": 4350.0, "data_quality": "good"}
    record["ai_outputs"] = {}
    record["gates"] = []
    record["fusion"] = {"calibrated_confidence": None}
    record["setup_quality"] = None
    record["conflicts"] = None
    record["sl"] = None
    text = TelegramNotifier(telegram_settings).proposal_message(
        _proposal(), record=record, proposal_id="pid-9"
    )
    assert "Biais MTF" not in text
    assert "Structure :" not in text
    assert "Gates :" not in text
    assert "Soutient :" not in text
    assert "Calibrée" not in text
    assert "Confiance brute : 0.72" in text
    assert "approve pid-9" in text


def test_legacy_fallback_without_record(telegram_settings):
    text = TelegramNotifier(telegram_settings).proposal_message(_proposal(), gauge=DXY_WEAK)
    assert "DXY : 60 (Bullish (USD weak))" in text
    assert "approve pid-1" in text
    assert "Risque : 50.00 $ | Confiance : 0.72" in text


def test_rejection_message_includes_reason_trace_and_analysis(telegram_settings):
    text = TelegramNotifier(telegram_settings).rejection_message(
        _record(decision_reason="dollar trop fort pour un LONG", no_trade_reason="DXY_FILTER")
    )
    assert "🚫 SIGNAL REJETÉ XAUUSD (15m)" in text
    assert "Raison : dollar trop fort pour un LONG (DXY_FILTER)" in text
    assert "Qualité des données : good" in text
    assert "Version : LEGACY_BASELINE" in text
    # The agents ran before the refusal: the WHY must be in the message.
    assert "Confiance brute : 0.70" in text
    assert "Gates : ai_signal ✓ | setup_quality ✓ | risk ✓" in text
    assert "TECHNICAL : long (0.80)" in text
    assert "REGIME : long (0.60) [heuristique]" in text
    assert "↳ r" in text  # agent reasoning snippet
    assert "Contredit : " in text and "volatilité faible (0.30)" in text


def test_rejection_message_labels_dxy_and_regime_payloads(telegram_settings):
    record = _record(decision_reason="x", no_trade_reason="LOW_CONFIDENCE")
    record["ai_outputs"] = {
        "dxy": {
            "agent": "dxy",
            "source": "llm",
            "payload": {"gold_bias": "short", "score": -0.4, "confidence": 0.7,
                        "reasoning": "dollar fort"},
        },
        "regime": {
            "agent": "regime",
            "source": "llm",
            "payload": {"regime": "ranging", "trend_direction": "flat",
                        "confidence": 0.55, "reasoning": "range"},
        },
    }
    text = TelegramNotifier(telegram_settings).rejection_message(record)
    assert "DXY : short (0.70)" in text
    assert "REGIME : flat (0.55)" in text
    assert "↳ dollar fort" in text


def test_rejection_message_concise_when_agents_did_not_run(telegram_settings):
    # Pre-AI refusals (news blackout, shock, data down) have no ai_outputs
    # and must stay concise — never invent an analysis that did not happen.
    record = {
        "signal_id": "sig-1",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "market_snapshot": {"data_quality": "good"},
        "decision_reason": "high-impact news within blackout window",
        "no_trade_reason": "NEWS_RISK",
    }
    text = TelegramNotifier(telegram_settings).rejection_message(record)
    assert "Raison : high-impact news within blackout window (NEWS_RISK)" in text
    assert "Qualité des données : good" in text
    assert "Version : LEGACY_BASELINE" in text
    assert "Confiance brute" not in text
    assert "Biais MTF" not in text
    assert "Gates :" not in text
    assert "TECHNICAL" not in text


def test_rejection_message_without_reason_has_fallback(telegram_settings):
    record = _record()
    record.pop("no_trade_reason", None)
    text = TelegramNotifier(telegram_settings).rejection_message(
        {"symbol": "XAUUSD", "market_snapshot": {}}
    )
    assert "Raison : aucune raison enregistrée" in text


def test_send_rejection_gated_off_when_disabled(monkeypatch):
    settings = Settings(
        telegram_bot_token="123:abc",
        telegram_chat_id="987",
        telegram_rejection_alerts=False,
    )
    sent: list[str] = []
    notifier = TelegramNotifier(settings)
    monkeypatch.setattr(notifier, "send", lambda text: sent.append(text) or True)
    assert not notifier.send_rejection(_record(decision_reason="x"))
    assert sent == []


def test_send_rejection_sends_when_enabled(monkeypatch):
    settings = Settings(
        telegram_bot_token="123:abc", telegram_chat_id="987", telegram_rejection_alerts=True
    )
    sent: list[str] = []
    notifier = TelegramNotifier(settings)
    monkeypatch.setattr(notifier, "send", lambda text: sent.append(text) or True)
    assert notifier.send_rejection(_record(decision_reason="x"))
    assert len(sent) == 1
    assert "SIGNAL REJETÉ" in sent[0]


def test_rejection_message_includes_timeframe_and_dedup_note(telegram_settings):
    text = TelegramNotifier(telegram_settings).rejection_message(
        _record(decision_reason="x", no_trade_reason="LOW_CONFIDENCE"),
        note="Même refus depuis 23:08 UTC — 4 cycles consécutifs",
    )
    assert "🚫 SIGNAL REJETÉ XAUUSD (15m)" in text
    assert "Même refus depuis 23:08 UTC — 4 cycles consécutifs" in text


def test_send_rejection_passes_note(monkeypatch):
    settings = Settings(
        telegram_bot_token="123:abc", telegram_chat_id="987", telegram_rejection_alerts=True
    )
    notifier = TelegramNotifier(settings)
    captured: dict = {}
    monkeypatch.setattr(
        notifier, "send", lambda text: captured.setdefault("text", text) or True
    )
    notifier.send_rejection(_record(decision_reason="x"), note="rappel")
    assert "rappel" in captured["text"]


def test_send_records_latency(telegram_settings, monkeypatch):
    import httpx

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: FakeResp())
    notifier = TelegramNotifier(telegram_settings)
    assert notifier.last_latency_ms == 0.0
    assert notifier.send("hi")
    assert notifier.last_latency_ms >= 0


def test_send_failure_still_records_latency(telegram_settings, monkeypatch):
    import httpx

    def boom(url, json=None, timeout=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(httpx, "post", boom)
    notifier = TelegramNotifier(telegram_settings)
    assert not notifier.send("hi")
    assert notifier.last_latency_ms >= 0


def test_send_signal_passes_record_and_proposal_id(telegram_settings, monkeypatch):
    captured: dict = {}
    notifier = TelegramNotifier(telegram_settings)

    def fake_send(text: str) -> bool:
        captured["text"] = text
        return True

    monkeypatch.setattr(notifier, "send", fake_send)
    notifier.send_signal(_proposal(), record=_record(), proposal_id="pid-7")
    assert "Version : LEGACY_BASELINE" in captured["text"]
    assert "approve pid-7" in captured["text"]
