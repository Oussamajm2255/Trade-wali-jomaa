"""Session classification (spec §11): ASIA / LONDON / OVERLAP / NY / OFF."""
from __future__ import annotations

from datetime import datetime, timezone

from trading_agent.data.sessions import session_context, session_start_utc


def utc(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


SUMMER_CASES = [  # Wed 2026-07-15: London 07:00-16:00 UTC, NY 13:30-21:00 UTC
    (utc(2026, 7, 15, 1, 0), "ASIA"),
    (utc(2026, 7, 15, 7, 0), "LONDON"),
    (utc(2026, 7, 15, 14, 0), "LONDON_NY_OVERLAP"),
    (utc(2026, 7, 15, 17, 0), "NEW_YORK"),
    (utc(2026, 7, 15, 21, 30), "OFF_SESSION"),
    (utc(2026, 7, 18, 2, 0), "OFF_SESSION"),  # Saturday: Asia closed too
]

WINTER_CASES = [  # Thu 2026-01-15: London 08:00-17:00 UTC, NY 14:30-22:00 UTC
    (utc(2026, 1, 15, 1, 0), "ASIA"),
    (utc(2026, 1, 15, 8, 0), "LONDON"),
    (utc(2026, 1, 15, 14, 30), "LONDON_NY_OVERLAP"),
    (utc(2026, 1, 15, 22, 30), "OFF_SESSION"),
]


def test_summer_classification():
    for now, expected in SUMMER_CASES:
        assert session_context(now)["session"] == expected, f"summer {now}"


def test_winter_classification():
    for now, expected in WINTER_CASES:
        assert session_context(now)["session"] == expected, f"winter {now}"


def test_overlap_flags():
    ctx = session_context(utc(2026, 7, 15, 14, 0))
    assert ctx["london"] and ctx["new_york"] and ctx["overlap"]
    ctx = session_context(utc(2026, 7, 15, 12, 0))
    assert ctx["london"] and not ctx["new_york"] and not ctx["overlap"]


def test_session_start_london():
    assert session_start_utc(utc(2026, 7, 15, 12, 0)) == utc(2026, 7, 15, 7, 0)


def test_session_start_overlap_is_ny_open():
    assert session_start_utc(utc(2026, 7, 15, 14, 0)) == utc(2026, 7, 15, 13, 30)


def test_session_start_ny_only_after_london_close():
    assert session_start_utc(utc(2026, 7, 15, 17, 30)) == utc(2026, 7, 15, 16, 0)


def test_session_start_asia():
    assert session_start_utc(utc(2026, 7, 15, 2, 0)) == utc(2026, 7, 15, 0, 0)


def test_session_start_off_session_is_day_start():
    assert session_start_utc(utc(2026, 7, 15, 22, 0)) == utc(2026, 7, 15, 0, 0)
