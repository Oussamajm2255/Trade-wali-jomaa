"""Post-signal forensics capture (V-MONSTER §65).

After every decision — proposal OR rejection — the market is sampled at
fixed offsets (1/3/5/10/30 minutes) and stored (`PostSnapshot`), so any
verdict can later be audited against how the setup actually aged.

The loop owns one `SnapshotRegistry` per process: entries register the
decision's signal_id + timestamp, `due()` returns the (signal, offset)
pairs whose window just elapsed, and `capture()` persists the current
candle. On a restart the registry restores itself from the last 31
minutes of stored decisions, so a brief downtime does not lose the
forensic trail (longer outages are honestly incomplete — spec §4).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from trading_agent.store import actions

logger = logging.getLogger(__name__)

# The spec-fixed sampling grid (minutes after the decision).
POST_OFFSETS_MIN = (1, 3, 5, 10, 30)

# Registry entries live only while captures are still possible.
_MAX_AGE_MIN = max(POST_OFFSETS_MIN) + 2


def _utc(dt: datetime) -> datetime:
    # SQLite returns naive datetimes; all arithmetic must stay aware.
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SnapshotRegistry:
    """Tracks recent decisions that still owe post-signal snapshots."""

    def __init__(self) -> None:
        self._entries: dict[str, dict] = {}

    def register(self, signal_id: str, ts: datetime, symbol: str) -> None:
        if not signal_id:
            return
        self._entries[signal_id] = {
            "ts": _utc(ts),
            "symbol": symbol,
            "captured": set(),
        }

    def restore(
        self, since_minutes: int = 31, now: datetime | None = None
    ) -> int:
        """Re-register recent decisions from the store (restart recovery)."""
        count = 0
        try:
            for row in actions.recent_decision_signals(since_minutes, now=now):
                if row["signal_id"] not in self._entries:
                    self.register(row["signal_id"], row["ts"], row["symbol"])
                    count += 1
        except Exception as exc:  # noqa: BLE001 - forensics must never kill the loop
            logger.warning("forensics restore failed: %s", exc)
        return count

    def due(self, now: datetime | None = None) -> list[tuple[str, int]]:
        """(signal_id, offset_min) pairs whose sampling window elapsed.

        Purely observational: nothing is marked. Call `mark()` once the
        snapshot is actually persisted, so a capture skipped for another
        symbol stays due on its own symbol's tick.
        """
        now = _utc(now) if now else datetime.now(timezone.utc)
        due: list[tuple[str, int]] = []
        for signal_id, entry in self._entries.items():
            elapsed_min = (now - entry["ts"]).total_seconds() / 60.0
            for offset in POST_OFFSETS_MIN:
                if offset not in entry["captured"] and elapsed_min >= offset:
                    due.append((signal_id, offset))
        self._prune(now)
        return due

    def mark(self, signal_id: str, offset: int) -> None:
        entry = self._entries.get(signal_id)
        if entry is not None:
            entry["captured"].add(offset)

    def _prune(self, now: datetime) -> None:
        for signal_id in list(self._entries):
            entry = self._entries[signal_id]
            if entry["captured"] and all(
                o in entry["captured"] for o in POST_OFFSETS_MIN
            ):
                self._entries.pop(signal_id, None)
            elif (now - entry["ts"]) > timedelta(minutes=_MAX_AGE_MIN):
                self._entries.pop(signal_id, None)

    def symbol_of(self, signal_id: str) -> str | None:
        entry = self._entries.get(signal_id)
        return entry["symbol"] if entry else None

    def __len__(self) -> int:
        return len(self._entries)


def capture_due(
    registry: SnapshotRegistry,
    symbol: str,
    candle,
    now: datetime | None = None,
) -> int:
    """Persist the snapshots this tick owes for one symbol's decisions.

    Returns the number of snapshots stored. Best-effort: a write
    failure is logged, never raised (observation path).
    """
    now = _utc(now) if now else datetime.now(timezone.utc)
    stored = 0
    for signal_id, offset in registry.due(now):
        if registry.symbol_of(signal_id) != symbol:
            continue
        try:
            inserted = actions.record_post_snapshot(
                signal_id,
                offset,
                float(candle["close"]),
                float(candle["high"]),
                float(candle["low"]),
                now=now,
            )
            # Mark either way: a row that already exists (restart
            # recovery) must not be retried every tick.
            registry.mark(signal_id, offset)
            if inserted:
                stored += 1
        except Exception as exc:  # noqa: BLE001 - capture must never kill the loop
            logger.warning("post snapshot failed for %s/%dm: %s", signal_id, offset, exc)
    return stored
