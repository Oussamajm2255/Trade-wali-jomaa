"""Narrative presentation layer tests (V-MONSTER §38 extension).

The narration is a pure function of the stored signal record. These
tests verify the non-negotiable guardrails:
- tone never stronger than the real confidence score (LOW/MEDIUM/HIGH);
- every NoTradeReason member is covered by a reason phrase;
- no fabricated numbers: every number in the narrative exists in the
  record;
- the watch level comes from stored liquidity/VWAP/stop only;
- the UNVALIDATED calibration is said out loud, and only when it is;
- the Telegram wiring keeps the full technical trace below the
  separator — nothing is deleted, only reorganized.
"""
from __future__ import annotations

import copy
import re

import pytest

from trading_agent.config import Settings
from trading_agent.fusion.types import NoTradeReason
from trading_agent.notify.narration import _REASON_PHRASES, build_narrative, tone_tier
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.schema.types import Side, SignalProposal

SEPARATOR = "── Détails (audit) ──"


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
            "dxy_gauge": {"value": 60, "classification": "Bullish (USD weak)", "kind": "dxy"},
            "session_context": {"session": "LONDON"},
            "price": 4350.0,
        },
        "ai_outputs": {
            "technical": {
                "agent": "technical",
                "source": "llm",
                "payload": {"bias": "long", "conviction": 0.8, "reasoning": "r"},
            }
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
        "gates": [{"gate": "ai_signal", "status": "pass"}],
        "final_decision": "proposal",
        "sl": 4300.0,
        "tp": 4450.0,
        "risk_amount": 50.0,
        "size": 1.2,
    }
    record.update(overrides)
    return record


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


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"-?\d[\d,]*\.\d{2}", text))


def _record_numbers(record: dict) -> set[str]:
    """Every float in the record, formatted exactly like the narration does."""
    found: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, float):
            found.add(f"{node:,.2f}")
        elif isinstance(node, int) and not isinstance(node, bool):
            found.add(f"{float(node):,.2f}")

    walk(record)
    return found


# ---------------------------------------------------------------- tone ladder


def test_tone_tier_thresholds():
    assert tone_tier({}) is None
    assert tone_tier({"raw_confidence": 0.12}) == "LOW"
    assert tone_tier({"raw_confidence": 0.5499}) == "LOW"
    assert tone_tier({"raw_confidence": 0.55}) == "MEDIUM"
    assert tone_tier({"raw_confidence": 0.7499}) == "MEDIUM"
    assert tone_tier({"raw_confidence": 0.75}) == "HIGH"
    # The Phase H statistical tier wins when stored.
    assert tone_tier({"tier": "HIGH", "raw_confidence": 0.1}) == "HIGH"
    assert tone_tier({"tier": "LOW", "raw_confidence": 0.9}) == "LOW"


def test_low_tone_never_sounds_confident():
    record = _record()
    record["fusion"] = {"raw_confidence": 0.12, "calibrated_confidence": None}
    text = build_narrative(record, side="LONG")
    assert "Je n'ai pas de conviction claire" in text
    assert "avec prudence" in text
    assert "avec conviction" not in text
    assert "conviction nette" not in text
    # UNVALIDATED confidence is said out loud.
    assert "pas encore validée" in text


def test_medium_tone_is_measured():
    record = _record()
    record["fusion"] = {"raw_confidence": 0.65, "calibrated_confidence": 0.6}
    text = build_narrative(record, side="LONG")
    assert "conviction mesurée" in text
    assert "avec conviction" not in text
    # Calibrated -> no UNVALIDATED hedge.
    assert "pas encore validée" not in text


def test_high_tone_is_clear_but_not_beyond_the_score():
    record = _record()
    record["fusion"] = {"raw_confidence": 0.8, "calibrated_confidence": 0.75}
    text = build_narrative(record, side="LONG")
    assert "le contexte est net" in text
    assert "avec conviction" in text
    assert "pas de conviction" not in text


# ------------------------------------------------------------ reason coverage


@pytest.mark.parametrize("reason", list(NoTradeReason))
def test_every_no_trade_reason_has_a_narrative(reason):
    record = {"no_trade_reason": reason.value, "market_snapshot": {}, "fusion": {}}
    text = build_narrative(record, rejected=True)
    assert text, f"no narrative for {reason.value}"
    assert "Je reste à l'écart" in text
    assert "À surveiller :" in text
    phrase = _REASON_PHRASES[reason.value]
    assert phrase in text or phrase.capitalize() in text


def test_conflicted_rejection_names_the_tension():
    record = _record(
        final_decision="rejected",
        no_trade_reason="STRUCTURE_CONFLICT",
        decision_reason="structure contre le LONG",
    )
    record["conflicts"] = {
        "state": "CONFLICTED",
        "conflicts": [{"axis": "structure", "detail": "structure baissière contre LONG 15m"}],
        "conflict_score": 0.8,
    }
    text = build_narrative(record, rejected=True)
    assert "Les signaux se contredisent nettement" in text
    assert "Je reste à l'écart" in text


def test_low_confidence_rejection_stays_humble():
    record = _record(final_decision="rejected", no_trade_reason="LOW_CONFIDENCE")
    record["fusion"] = {"raw_confidence": 0.259}
    text = build_narrative(record, rejected=True)
    assert "Je n'ai pas de conviction claire" in text
    assert "Je reste à l'écart" in text
    assert "conviction nette" not in text


def test_high_tier_rejection_admits_potential_but_stays_out():
    record = _record(final_decision="rejected", no_trade_reason="DXY_CONFLICT")
    record["fusion"] = {"raw_confidence": 0.8, "calibrated_confidence": 0.75}
    text = build_narrative(record, rejected=True)
    assert "Le setup avait du potentiel" in text
    assert "la discipline prime" in text
    assert "Je reste à l'écart" in text
    assert "pas de conviction" not in text


# -------------------------------------------------------- no fabricated data


def test_no_fabricated_numbers():
    record = _record()
    record["market_snapshot"]["liquidity"] = {
        "nearest_above": {"price": 4261.5, "kind": "equal highs"},
        "nearest_below": {"price": 4233.0, "kind": "equal lows"},
        "quality": 0.8,
    }
    text = build_narrative(record, side="LONG")
    assert _numbers(text) <= _record_numbers(record)


def test_watch_level_comes_from_real_liquidity():
    record = _record()
    record["market_snapshot"]["liquidity"] = {
        "nearest_above": {"price": 4261.5, "kind": "equal highs"},
        "nearest_below": {"price": 4233.0, "kind": "equal lows"},
    }
    text = build_narrative(record, side="LONG")
    assert "4,261.50" in text
    assert "equal highs" in text
    assert "4,300.00 invalide le setup" in text


def test_watch_level_falls_back_honestly():
    record = _record(sl=None, tp=None)
    record["market_snapshot"].pop("liquidity", None)
    text = build_narrative(record, side="LONG")
    assert "Aucun niveau calculé" in text


def test_tick_volume_vwap_is_labelled_in_narrative():
    record = _record(sl=None, tp=None)
    record["market_snapshot"]["vwap"] = {
        "available": True,
        "session_vwap": 4352.1,
        "volume_basis": "tick",
    }
    text = build_narrative(record, side="LONG")
    assert "4,352.10 (volume tick)" in text
    record["market_snapshot"]["vwap"]["volume_basis"] = "real"
    text = build_narrative(record, side="LONG")
    assert "(volume tick)" not in text
    assert "4,352.10" in text


def test_narrative_is_deterministic():
    record = _record()
    first = build_narrative(record, side="LONG")
    second = build_narrative(copy.deepcopy(record), side="LONG")
    third = build_narrative(record, side="LONG")
    assert first == second == third


# ------------------------------------------------------------ telegram wiring


def test_proposal_message_narrative_above_details():
    notifier = TelegramNotifier(
        Settings(telegram_bot_token="123:abc", telegram_chat_id="987")
    )
    text = notifier.proposal_message(_proposal(), record=_record(), proposal_id="pid-9")
    assert SEPARATOR in text
    narrative_part = text.split(SEPARATOR)[0]
    details_part = text.split(SEPARATOR)[1]
    # The narrative is human prose, not a debug dump.
    assert "conviction" in narrative_part
    assert "Soutient :" not in narrative_part
    assert "Confiance brute" not in narrative_part
    # Every technical section survives below the separator.
    assert "Entrée : 4,350.00" in details_part
    assert "Confiance brute : 0.72" in details_part
    assert "Soutient :" in details_part
    assert "approve pid-9" in details_part


def test_rejection_message_narrative_above_details():
    notifier = TelegramNotifier(
        Settings(telegram_bot_token="123:abc", telegram_chat_id="987")
    )
    record = _record(final_decision="rejected", no_trade_reason="LOW_CONFIDENCE")
    text = notifier.rejection_message(record)
    assert SEPARATOR in text
    narrative_part = text.split(SEPARATOR)[0]
    assert "Je reste à l'écart" in narrative_part
    assert "propose" not in narrative_part
    assert "Raison :" in text.split(SEPARATOR)[1]
    assert "Confiance brute : 0.70" in text.split(SEPARATOR)[1]


def test_pre_ai_refusal_narrative_stays_concise():
    notifier = TelegramNotifier(
        Settings(telegram_bot_token="123:abc", telegram_chat_id="987")
    )
    record = {
        "signal_id": "sig-1",
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "market_snapshot": {"data_quality": "good"},
        "decision_reason": "high-impact news within blackout window",
        "no_trade_reason": "NEWS_RISK",
    }
    text = notifier.rejection_message(record)
    assert SEPARATOR in text
    narrative_part = text.split(SEPARATOR)[0]
    assert "news" in narrative_part
    assert "Je reste à l'écart" in narrative_part
    # Nothing invented about an analysis that never ran.
    assert "Confiance brute" not in text
    assert "TECHNICAL" not in text
    assert "Raison : high-impact news within blackout window (NEWS_RISK)" in text


def test_narrative_can_be_disabled():
    settings = Settings(
        telegram_bot_token="123:abc",
        telegram_chat_id="987",
        telegram_narrative_enabled=False,
    )
    notifier = TelegramNotifier(settings)
    text = notifier.proposal_message(_proposal(), record=_record(), proposal_id="pid-9")
    assert SEPARATOR not in text
    assert "Entrée : 4,350.00" in text
    rejection = notifier.rejection_message(
        _record(final_decision="rejected", no_trade_reason="LOW_CONFIDENCE")
    )
    assert SEPARATOR not in rejection
    assert "Raison :" in rejection


def test_full_message_is_deterministic():
    notifier = TelegramNotifier(
        Settings(telegram_bot_token="123:abc", telegram_chat_id="987")
    )
    assert notifier.proposal_message(
        _proposal(), record=_record(), proposal_id="pid-9"
    ) == notifier.proposal_message(_proposal(), record=_record(), proposal_id="pid-9")
