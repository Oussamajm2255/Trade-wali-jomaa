"""Economic calendar interface (spec §43).

Events only ever come from a REAL provider — the LLM is never asked
about future news. Events are normalized to one shape:

    {"event": "CPI", "currency": "USD", "importance": "HIGH",
     "minutes_to_event": 32}

The default provider is the null provider (offline-safe, zero events):
no events means no blocking, never a fabricated one. Providers fail
open — a broken calendar must not kill the analysis cycle.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from trading_agent.config import Settings

logger = logging.getLogger(__name__)

# Finnhub `impact` values -> spec §43 importance.
_IMPACT_TO_IMPORTANCE = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
# Finnhub `country` codes for the dollar side of XAUUSD.
_USD_COUNTRIES = {"US", "USA", "United States"}

_CACHE_TTL_SECONDS = 300  # one light fetch per 5 minutes at most


class EconomicCalendarProvider(Protocol):
    """Anything that can list upcoming normalized economic events."""

    def upcoming_events(self, now: datetime, window_minutes: int) -> list[dict]: ...


class NullCalendarProvider:
    """Offline default: no events, no news blocking (spec §43 optional)."""

    def upcoming_events(self, now: datetime, window_minutes: int) -> list[dict]:
        return []


class FinnhubCalendarProvider:
    """Real calendar data from the free Finnhub economic-calendar API.

    Fails open: network/API problems return [] with a warning — the
    robot must never invent events, and must never crash on a broken
    calendar feed.
    """

    ENDPOINT = "https://finnhub.io/api/v1/calendar/economic"

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._cache: tuple[float, list[dict]] | None = None

    def upcoming_events(self, now: datetime, window_minutes: int) -> list[dict]:
        if not self.api_key:
            return []
        events = self._cached(now)
        return [e for e in events if 0 <= e["minutes_to_event"] <= window_minutes]

    def _cached(self, now: datetime) -> list[dict]:
        if self._cache and now.timestamp() - self._cache[0] < _CACHE_TTL_SECONDS:
            return self._cache[1]
        events = self._fetch(now)
        self._cache = (now.timestamp(), events)
        return events

    def _fetch(self, now: datetime) -> list[dict]:
        import requests

        start = now.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                self.ENDPOINT,
                params={"from": start, "to": start, "token": self.api_key},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - fail open, log the reason
            logger.warning("economic calendar fetch failed: %s", exc)
            return []
        raw = payload.get("economicCalendar") or []
        return normalize_events(raw, now)


def normalize_events(raw: list[dict], now: datetime) -> list[dict]:
    """Map raw provider rows to spec §43 events (USD-only, importance
    ranked, minutes_to_event computed from the provider's UTC time)."""
    events: list[dict] = []
    for row in raw:
        country = str(row.get("country") or "").strip()
        if country not in _USD_COUNTRIES:
            continue
        importance = _IMPACT_TO_IMPORTANCE.get(str(row.get("impact") or "").lower())
        if importance is None:
            continue
        minutes = _minutes_until(row, now)
        if minutes is None:
            continue
        events.append(
            {
                "event": str(row.get("event") or "").strip(),
                "currency": "USD",
                "importance": importance,
                "minutes_to_event": minutes,
            }
        )
    return events


def _minutes_until(row: dict, now: datetime) -> int | None:
    """Event UTC datetime from the provider fields -> minutes until it.

    Finnhub puts `date` (YYYY-MM-DD) and `time` (HH:MM:SS) side by
    side; anything unparseable is skipped rather than guessed.
    """
    date = str(row.get("date") or "").strip()
    time = str(row.get("time") or "").strip()
    if not date:
        return None
    text = f"{date}T{time or '00:00:00'}"
    try:
        ts = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int((ts - now).total_seconds() // 60)


_IMPORTANCE_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def blocking_events(
    events: list[dict],
    min_importance: str,
    block_minutes: int,
) -> list[dict]:
    """The subset of events that block new entries (spec §43)."""
    rank = _IMPORTANCE_RANK.get(str(min_importance).upper(), 0)
    return [
        e
        for e in events
        if _IMPORTANCE_RANK.get(str(e.get("importance") or "").upper(), 0) >= rank
        and 0 <= e.get("minutes_to_event", 10**9) <= block_minutes
    ]


def build_calendar_provider(settings: Settings) -> EconomicCalendarProvider:
    """Provider per config; anything unknown falls back to null (safe)."""
    if settings.news_provider == "finnhub":
        return FinnhubCalendarProvider(settings.finnhub_api_key)
    return NullCalendarProvider()
