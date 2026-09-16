"""Session filter: London/NY/Sydney windows, DST-aware (summer and winter)."""
from __future__ import annotations

from datetime import datetime, timezone

from trading_agent.data.sessions import (
    next_session_open,
    session_context,
    session_state,
)


def utc(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# Summer (BST/EDT/AEST): London 08:00-17:00 local = 07:00-16:00 UTC,
# NY 09:30-17:00 local = 13:30-21:00 UTC,
# Sydney 07:00-16:00 AEST = 21:00-06:00 UTC.
SUMMER = [
    (utc(2026, 7, 15, 6, 0), False),   # Sydney closed, before London
    (utc(2026, 7, 15, 7, 0), True),    # London open (08:00 BST)
    (utc(2026, 7, 15, 12, 0), True),   # London only
    (utc(2026, 7, 15, 13, 30), True),  # overlap start (09:30 EDT)
    (utc(2026, 7, 15, 16, 0), True),   # London closed, NY open
    (utc(2026, 7, 15, 21, 0), True),   # NY closed, Sydney open (07:00 AEST)
    (utc(2026, 7, 15, 23, 0), True),   # Sydney
]

# Winter (GMT/EST/AEDT): London 08:00-17:00 UTC, NY 09:30-17:00 = 14:30-22:00 UTC,
# Sydney 07:00-16:00 AEDT = 20:00-05:00 UTC.
WINTER = [
    (utc(2026, 1, 15, 7, 30), False),  # Sydney closed (05:00 AEDT end), London closed
    (utc(2026, 1, 15, 8, 0), True),    # London open
    (utc(2026, 1, 15, 14, 0), True),   # London only (09:00 EST)
    (utc(2026, 1, 15, 14, 30), True),  # overlap start
    (utc(2026, 1, 15, 17, 30), True),  # NY only
    (utc(2026, 1, 15, 22, 0), True),   # NY closed, Sydney open (09:00 AEDT)
]


def test_summer_windows():
    for now, expected in SUMMER:
        assert session_state(now)["in_session"] is expected, f"summer {now}"


def test_winter_windows():
    for now, expected in WINTER:
        assert session_state(now)["in_session"] is expected, f"winter {now}"


def test_overlap_detection():
    state = session_state(utc(2026, 7, 15, 14, 0))  # 10:00 EDT, 15:00 BST
    assert state["overlap"] is True
    assert state["london"] and state["new_york"]
    state = session_state(utc(2026, 7, 15, 12, 0))  # London only
    assert state["london"] and not state["new_york"]
    assert not state["overlap"]


def test_next_open_overnight():
    # Sydney closed at 06:00 UTC, London opens 07:00 UTC the same day.
    nxt = next_session_open(utc(2026, 7, 15, 6, 30))
    assert nxt == utc(2026, 7, 15, 7, 0)


def test_next_open_weekend():
    # Saturday afternoon -> Sydney opens the FX week on Sunday 21:00 UTC
    # (Monday 07:00 AEST).
    nxt = next_session_open(utc(2026, 7, 18, 15, 0))
    assert nxt == utc(2026, 7, 19, 21, 0)


def test_naive_datetime_treated_as_utc():
    naive = datetime(2026, 7, 15, 12, 0)
    assert session_state(naive)["in_session"] is True


# ------------------------------------------------------------- Sydney


def test_sydney_windows_dst_aware():
    # July (AEST, UTC+10): 21:00-06:00 UTC.
    assert session_state(utc(2026, 7, 15, 23, 0))["sydney"] is True
    assert session_state(utc(2026, 7, 15, 6, 0))["sydney"] is False  # 16:00 AEST end
    # January (AEDT, UTC+11): 20:00-05:00 UTC.
    assert session_state(utc(2026, 1, 15, 20, 30))["sydney"] is True
    assert session_state(utc(2026, 1, 15, 5, 30))["sydney"] is False


def test_sydney_weekend_closed():
    # Saturday 23:00 UTC = Sunday 09:00 AEST -> no session at all.
    state = session_state(utc(2026, 7, 18, 23, 0))
    assert state["sydney"] is False
    assert state["in_session"] is False


def test_sydney_disabled_with_empty_window():
    state = session_state(utc(2026, 7, 15, 23, 0), sydney="")
    assert state["sydney"] is False
    assert state["in_session"] is False


def test_sydney_classification_label():
    # 23:00 UTC: London/NY/Tokyo closed, Sydney open -> SYDNEY.
    ctx = session_context(utc(2026, 7, 15, 23, 0))
    assert ctx["session"] == "SYDNEY"
    assert ctx["sydney"] is True
    # 23:00 UTC with Sydney disabled: no window is open -> OFF_SESSION.
    ctx = session_context(utc(2026, 7, 15, 23, 0), sydney="")
    assert ctx["session"] == "OFF_SESSION"
