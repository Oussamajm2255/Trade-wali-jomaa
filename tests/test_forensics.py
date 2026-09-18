"""Post-signal forensics (V-MONSTER §65): snapshot registry + capture."""

from datetime import datetime, timedelta, timezone

from trading_agent.forensics import (
    POST_OFFSETS_MIN,
    SnapshotRegistry,
    capture_due,
)
from trading_agent.store import actions

NOW = datetime(2026, 9, 17, 10, 0, tzinfo=timezone.utc)


def _candle(close=4350.0, high=4351.5, low=4348.5):
    return {"close": close, "high": high, "low": low}


def _signal(signal_id, ts, decision="proposal", symbol="XAUUSD", **overrides):
    row = {
        "signal_id": signal_id,
        "ts": ts,
        "symbol": symbol,
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "config_version": "v1",
        "prompt_version": "p1",
        "market_snapshot": {"price": 4350.0},
        "ai_outputs": {},
        "fusion": {"direction_score": 0.6},
        "setup_quality": {"score": 0.7},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [],
        "final_decision": decision,
        "outcome": None,
        "r_multiple": None,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------- registry timing


def test_due_only_returns_elapsed_offsets():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW, "XAUUSD")
    assert reg.due(NOW + timedelta(seconds=30)) == []
    assert reg.due(NOW + timedelta(minutes=1)) == [("sig1", 1)]
    assert reg.due(NOW + timedelta(minutes=3, seconds=1)) == [
        ("sig1", 1),
        ("sig1", 3),
    ]
    assert reg.due(NOW + timedelta(minutes=31)) == list(
        (("sig1", o) for o in POST_OFFSETS_MIN)
    )


def test_due_is_observational_until_mark():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW, "XAUUSD")
    at_5m = NOW + timedelta(minutes=5)
    first = reg.due(at_5m)
    assert first == [("sig1", 1), ("sig1", 3), ("sig1", 5)]
    # due() alone must not consume anything (a capture skipped for
    # another symbol stays due on its own symbol's tick).
    assert reg.due(at_5m) == first


def test_mark_consumes_an_offset():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW, "XAUUSD")
    reg.mark("sig1", 1)
    # At 2 minutes only offset 1 would have been due — it is consumed.
    assert reg.due(NOW + timedelta(minutes=2)) == []


def test_mark_unknown_id_is_silent():
    reg = SnapshotRegistry()
    reg.mark("missing", 1)  # must not raise
    assert len(reg) == 0


def test_entry_pruned_when_fully_captured():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW, "XAUUSD")
    for o in POST_OFFSETS_MIN:
        reg.mark("sig1", o)
    reg.due(NOW + timedelta(minutes=2))
    assert len(reg) == 0


def test_entry_pruned_after_max_age():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW, "XAUUSD")
    reg.due(NOW + timedelta(minutes=33))
    assert len(reg) == 0
    assert reg.due(NOW + timedelta(minutes=34)) == []


def test_naive_ts_is_normalized():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW.replace(tzinfo=None), "XAUUSD")  # SQLite-style naive
    assert reg.due(NOW + timedelta(minutes=1)) == [("sig1", 1)]


def test_symbol_of_returns_registered_symbol():
    reg = SnapshotRegistry()
    reg.register("sig1", NOW, "XAUUSD")
    assert reg.symbol_of("sig1") == "XAUUSD"
    assert reg.symbol_of("missing") is None


# ------------------------------------------------------------- restore path


def test_restore_recovers_recent_decisions():
    actions.record_signal(_signal("sigA", NOW - timedelta(minutes=10)))
    actions.record_signal(_signal("sigB", NOW - timedelta(minutes=40)))
    reg = SnapshotRegistry()
    assert reg.restore(since_minutes=31, now=NOW) == 1
    assert reg.symbol_of("sigA") == "XAUUSD"
    assert reg.symbol_of("sigB") is None


def test_restore_never_overwrites_live_entries():
    actions.record_signal(_signal("sigA", NOW - timedelta(minutes=10)))
    reg = SnapshotRegistry()
    reg.register("sigA", NOW, "XAUUSD")
    reg.mark("sigA", 1)
    assert reg.restore(since_minutes=31, now=NOW) == 0
    assert reg.due(NOW + timedelta(minutes=2)) == []


# ------------------------------------------------------------- capture path


def test_capture_due_persists_and_returns_inserted_count():
    reg = SnapshotRegistry()
    reg.register("sigA", NOW - timedelta(minutes=1), "XAUUSD")
    stored = capture_due(reg, "XAUUSD", _candle(), now=NOW)
    assert stored == 1
    snaps = actions.post_snapshots("sigA")
    assert [s["offset_min"] for s in snaps] == [1]
    assert snaps[0]["price"] == 4350.0
    assert snaps[0]["high"] == 4351.5
    assert snaps[0]["low"] == 4348.5


def test_capture_due_respects_symbol():
    reg = SnapshotRegistry()
    reg.register("sigA", NOW - timedelta(minutes=1), "XAUUSD")
    reg.register("sigB", NOW - timedelta(minutes=1), "EURUSD")
    # A tick for XAUUSD must not consume EURUSD's entry, and vice versa.
    assert capture_due(reg, "XAUUSD", _candle(), now=NOW) == 1
    assert reg.due(NOW) == [("sigB", 1)]
    assert actions.post_snapshots("sigB") == []


def test_capture_due_marks_existing_rows_without_retry():
    actions.record_post_snapshot("sigA", 1, 4350.0, 4351.0, 4349.0, now=NOW)
    reg = SnapshotRegistry()
    reg.register("sigA", NOW - timedelta(minutes=1), "XAUUSD")
    # Row already persisted (restart recovery): not inserted, but the
    # offset must still be consumed so it is not retried every tick.
    assert capture_due(reg, "XAUUSD", _candle(), now=NOW) == 0
    assert reg.due(NOW) == []
    assert len(actions.post_snapshots("sigA")) == 1


def test_record_post_snapshot_is_idempotent():
    assert actions.record_post_snapshot("s", 1, 1.0, 1.0, 1.0, now=NOW) is True
    assert actions.record_post_snapshot("s", 1, 2.0, 2.0, 2.0, now=NOW) is False
    snaps = actions.post_snapshots("s")
    assert len(snaps) == 1
    assert snaps[0]["price"] == 1.0  # first write wins


def test_post_snapshots_ordered_by_offset():
    for off in (30, 1, 10):
        actions.record_post_snapshot("s", off, 1.0, 1.0, 1.0, now=NOW)
    assert [s["offset_min"] for s in actions.post_snapshots("s")] == [1, 10, 30]


def test_recent_decision_signals_window():
    actions.record_signal(_signal("sigA", NOW - timedelta(minutes=5)))
    actions.record_signal(_signal("sigB", NOW - timedelta(minutes=60)))
    rows = actions.recent_decision_signals(since_minutes=31, now=NOW)
    assert [r["signal_id"] for r in rows] == ["sigA"]
    assert rows[0]["symbol"] == "XAUUSD"
