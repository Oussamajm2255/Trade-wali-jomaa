"""Trading sessions — London, New York and Asia, DST-aware.

Two roles (spec §11):

1. session_state() — the analysis GATE: whether London or New York is
   open right now (kept separate; a session label is stored with every
   signal, but no session is forced better than another).
2. session_context() — the classification: ASIA / LONDON /
   LONDON_NY_OVERLAP / NEW_YORK / OFF_SESSION, so historical
   performance can later be measured per session.

Window times are LOCAL times; zoneinfo applies summer/winter offsets
automatically, so nothing to adjust in March or October.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta, timezone

from zoneinfo import ZoneInfo

LONDON_TZ = ZoneInfo("Europe/London")
NEW_YORK_TZ = ZoneInfo("America/New_York")
ASIA_TZ = ZoneInfo("Asia/Tokyo")

DEFAULT_LONDON = "08:00-17:00"  # local time
DEFAULT_NEW_YORK = "09:30-17:00"  # local time
DEFAULT_ASIA = "09:00-18:00"  # Tokyo local — the quiet overnight gold window


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


def _asia_open(now: datetime, window: str) -> bool:
    """Tokyo-session window open at UTC `now` (Mon-Fri only)."""
    start, end = _parse_window(window)
    local = now.astimezone(ASIA_TZ)
    return local.weekday() < 5 and start <= local.time() < end


def session_context(
    now: datetime | None = None,
    london: str = DEFAULT_LONDON,
    new_york: str = DEFAULT_NEW_YORK,
    asia: str = DEFAULT_ASIA,
) -> dict:
    """Session classification (spec §11): ASIA / LONDON / LONDON_NY_OVERLAP /
    NEW_YORK / OFF_SESSION. Informational only — never a forced gate; the
    label is stored with every signal for per-session performance analytics."""
    now = _utc(now)
    london_open, ny_open = _windows_open(now, london, new_york)
    asia_open = _asia_open(now, asia)
    if london_open and ny_open:
        label = "LONDON_NY_OVERLAP"
    elif ny_open:
        label = "NEW_YORK"
    elif london_open:
        label = "LONDON"
    elif asia_open:
        label = "ASIA"
    else:
        label = "OFF_SESSION"
    return {
        "session": label,
        "london": london_open,
        "new_york": ny_open,
        "asia": asia_open,
        "overlap": london_open and ny_open,
    }


def session_start_utc(
    now: datetime | None = None,
    london: str = DEFAULT_LONDON,
    new_york: str = DEFAULT_NEW_YORK,
    asia: str = DEFAULT_ASIA,
) -> datetime:
    """Start (UTC) of the classified session currently in progress.

    Follows session_context()'s labels: the start of the LONDON window
    during LONDON, the start of the NY window during LONDON_NY_OVERLAP /
    NEW_YORK, the start of the ASIA window during ASIA. OFF_SESSION
    returns the start of the current UTC day (intraday fallback).
    """
    now = _utc(now)
    label = session_context(now, london, new_york, asia)["session"]
    if label == "OFF_SESSION":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    probe = now
    for _ in range(26 * 60):  # scan back up to 26h while the label persists
        if session_context(probe, london, new_york, asia)["session"] != label:
            return probe + timedelta(minutes=1)
        probe -= timedelta(minutes=1)
    return probe  # unreachable with sane windows
