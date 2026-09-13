"""Trading session filter — London & New York, DST-aware.

Analysis runs only while London or New York is open: the overnight Asian
hours produce drift, not signal, for XAUUSD, so the robot sleeps instead
of spending LLM calls on noise. Window times are LOCAL times; zoneinfo
handles summer/winter offsets automatically, so nothing to adjust in
March or October. Position management stays 24/7 — this module only
gates analysis.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, timezone

from zoneinfo import ZoneInfo

LONDON_TZ = ZoneInfo("Europe/London")
NEW_YORK_TZ = ZoneInfo("America/New_York")

DEFAULT_LONDON = "08:00-17:00"  # local time
DEFAULT_NEW_YORK = "09:30-17:00"  # local time


def _parse_window(window: str) -> tuple[dtime, dtime]:
    start_s, end_s = window.split("-")
    hh, mm = start_s.strip().split(":")
    start = dtime(int(hh), int(mm))
    hh, mm = end_s.strip().split(":")
    end = dtime(int(hh), int(mm))
    return start, end


def _utc(now: datetime | None) -> datetime:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _windows_open(now: datetime, london: str, new_york: str) -> tuple[bool, bool]:
    """(london_open, ny_open) at UTC `now`. Monday-Friday only: the
    London/NY sessions do not exist on weekends."""
    london_start, london_end = _parse_window(london)
    ny_start, ny_end = _parse_window(new_york)
    london_local = now.astimezone(LONDON_TZ)
    ny_local = now.astimezone(NEW_YORK_TZ)
    london_open = (
        london_local.weekday() < 5
        and london_start <= london_local.time() < london_end
    )
    ny_open = ny_local.weekday() < 5 and ny_start <= ny_local.time() < ny_end
    return london_open, ny_open


def session_state(
    now: datetime | None = None,
    london: str = DEFAULT_LONDON,
    new_york: str = DEFAULT_NEW_YORK,
) -> dict:
    """Return the session status at `now`, plus the next opening time (UTC).

    Keys: in_session, london, new_york, overlap, next_open (ISO 8601 UTC).
    """
    now = _utc(now)
    london_open, ny_open = _windows_open(now, london, new_york)
    return {
        "in_session": london_open or ny_open,
        "london": london_open,
        "new_york": ny_open,
        "overlap": london_open and ny_open,
        "next_open": next_session_open(now, london, new_york).isoformat(),
    }


def next_session_open(
    now: datetime | None = None,
    london: str = DEFAULT_LONDON,
    new_york: str = DEFAULT_NEW_YORK,
) -> datetime:
    """First future time (UTC) when London or New York is open."""
    probe = _utc(now) + timedelta(minutes=1)
    for _ in range(48 * 60):  # scan up to 48h in 1-minute steps
        london_open, ny_open = _windows_open(probe, london, new_york)
        if london_open or ny_open:
            return probe
        probe += timedelta(minutes=1)
    return probe  # unreachable with sane windows
