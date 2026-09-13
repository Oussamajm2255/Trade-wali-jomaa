"""Session filter: London/NY windows, DST-aware (summer and winter)."""
from __future__ import annotations

from datetime import datetime, timezone

from trading_agent.data.sessions import next_session_open, session_state


def utc(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# Summer (BST/EDT): London 08:00-17:00 local = 07:00-16:00 UTC,
# NY 09:30-17:00 local = 13:30-21:00 UTC.
SUMMER = [
    (utc(2026, 7, 15, 6, 0), False),   # before London
    (utc(2026, 7, 15, 7, 0), True),    # London open (08:00 BST)
    (utc(2026, 7, 15, 12, 0), True),   # London only
    (utc(2026, 7, 15, 13, 30), True),  # overlap start (09:30 EDT)
    (utc(2026, 7, 15, 16, 0), True),   # London closed, NY open
    (utc(2026, 7, 15, 21, 0), False),  # NY closed (17:00 EDT)
    (utc(2026, 7, 15, 23, 0), False),  # overnight
]

# Winter (GMT/EST): London 08:00-17:00 UTC, NY 09:30-17:00 = 14:30-22:00 UTC.
WINTER = [
    (utc(2026, 1, 15, 7, 30), False),
    (utc(2026, 1, 15, 8, 0), True),    # London open
    (utc(2026, 1, 15, 14, 0), True),   # London only (09:00 EST)
    (utc(2026, 1, 15, 14, 30), True),  # overlap start
    (utc(2026, 1, 15, 17, 30), True),  # NY only
    (utc(2026, 1, 15, 22, 0), False),  # NY closed
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
    nxt = next_session_open(utc(2026, 7, 15, 23, 0))
    assert nxt == utc(2026, 7, 16, 7, 0)  # London open next day


def test_next_open_weekend():
    # Saturday afternoon -> Monday London open (UTC+1 in July).
    nxt = next_session_open(utc(2026, 7, 18, 15, 0))
    assert nxt == utc(2026, 7, 20, 7, 0)


def test_naive_datetime_treated_as_utc():
    naive = datetime(2026, 7, 15, 12, 0)
    assert session_state(naive)["in_session"] is True
