"""Dedup guard for rejection phone alerts (§38 + trader request).

First refusal of a gate code sends immediately; repeats are suppressed
until the heartbeat window elapses; a code change re-arms instantly and
carries a summary of the suppressed run it ended.
"""

from datetime import datetime, timezone

from trading_agent.notify.dedup import RejectionDedup

NOW = datetime(2026, 9, 17, 23, 8, tzinfo=timezone.utc)


def test_first_occurrence_sends_without_note():
    d = RejectionDedup(repeat_minutes=60)
    send, note = d.should_send("LOW_CONFIDENCE", now=NOW, mono=1000.0)
    assert send and note is None


def test_consecutive_same_code_suppressed():
    d = RejectionDedup(repeat_minutes=60)
    d.should_send("LOW_CONFIDENCE", mono=1000.0)
    assert d.should_send("LOW_CONFIDENCE", mono=1001.0) == (False, None)
    assert d.should_send("LOW_CONFIDENCE", mono=1002.0) == (False, None)


def test_heartbeat_resend_after_repeat_minutes():
    d = RejectionDedup(repeat_minutes=60)
    d.should_send("LOW_CONFIDENCE", now=NOW, mono=1000.0)
    assert not d.should_send("LOW_CONFIDENCE", mono=1001.0)[0]
    send, note = d.should_send("LOW_CONFIDENCE", now=NOW, mono=1000.0 + 3600)
    assert send
    assert "23:08 UTC" in note
    assert "3 cycles consécutifs" in note


def test_code_change_rearms_immediately_with_run_summary():
    d = RejectionDedup(repeat_minutes=60)
    d.should_send("LOW_CONFIDENCE", now=NOW, mono=1000.0)
    d.should_send("LOW_CONFIDENCE", mono=1001.0)
    d.should_send("LOW_CONFIDENCE", mono=1002.0)
    send, note = d.should_send("HTF_BIAS", now=NOW, mono=1003.0)
    assert send
    assert "LOW_CONFIDENCE ×3" in note
    assert "23:08 UTC" in note


def test_heartbeat_count_includes_first_occurrence():
    d = RejectionDedup(repeat_minutes=60)
    d.should_send("SHOCK", now=NOW, mono=0.0)
    _, note = d.should_send("SHOCK", now=NOW, mono=3600.0)
    assert "2 cycles consécutifs" in note


def test_unknown_codes_never_raise():
    d = RejectionDedup(repeat_minutes=60)
    assert d.should_send(None, mono=0.0)[0]  # type: ignore[arg-type]
    assert not d.should_send(None, mono=1.0)[0]  # type: ignore[arg-type]
