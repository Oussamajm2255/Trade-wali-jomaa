"""MTF alignment classification (spec §5): BULLISH/BEARISH/MIXED/CONFLICTED."""
from __future__ import annotations

from trading_agent.data.alignment import classify_alignment


def test_all_bull_alignment() -> None:
    biases = {tf: {"bias": "bull"} for tf in ("1d", "4h", "1h", "15m")}
    r = classify_alignment(biases)
    assert r["alignment"] == "BULLISH_ALIGNMENT"
    assert r["alignment_score"] == 1.0
    assert r["bull_tfs"] == 4 and r["bear_tfs"] == 0


def test_all_bear_alignment() -> None:
    biases = {tf: {"bias": "bear"} for tf in ("1d", "4h", "1h", "15m")}
    r = classify_alignment(biases)
    assert r["alignment"] == "BEARISH_ALIGNMENT"
    assert r["alignment_score"] == -1.0


def test_conflicted_when_bull_and_bear_timeframes() -> None:
    biases = {
        "1d": {"bias": "bull"},
        "4h": {"bias": "bull"},
        "1h": {"bias": "bear"},
        "15m": {"bias": "bear"},
    }
    r = classify_alignment(biases)
    assert r["alignment"] == "CONFLICTED"
    assert r["alignment_score"] == 0.0
    assert r["bull_tfs"] == 2 and r["bear_tfs"] == 2


def test_single_directional_timeframe_is_mixed() -> None:
    biases = {
        "1d": {"bias": "neutral"},
        "4h": {"bias": "neutral"},
        "1h": {"bias": "neutral"},
        "15m": {"bias": "bull"},
    }
    r = classify_alignment(biases)
    assert r["alignment"] == "MIXED"
    assert r["alignment_score"] == 1.0  # direction agreement, not enough TFs


def test_no_directional_bias_is_mixed() -> None:
    r = classify_alignment({tf: {"bias": "neutral"} for tf in ("1d", "4h", "1h", "15m")})
    assert r["alignment"] == "MIXED"
    assert r["alignment_score"] == 0.0
    assert r["bull_tfs"] == 0 and r["bear_tfs"] == 0


def test_missing_timeframes_treated_as_neutral() -> None:
    r = classify_alignment({"4h": {"bias": "bull"}, "1h": {"bias": "bull"}})
    assert r["alignment"] == "BULLISH_ALIGNMENT"
