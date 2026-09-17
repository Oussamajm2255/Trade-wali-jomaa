"""Consecutive-rejection spam guard for Telegram alerts.

Gate reasons are value-parameterised ("confidence 0.51 below minimum
0.55"), so two refusals by the SAME gate almost never share an identical
string. Deduplication therefore keys on the NO-TRADE gate code
(LOW_CONFIDENCE, HTF_BIAS, ...), falling back to the raw reason when a
rejection carries no code.

Behaviour: the first occurrence of a code sends immediately; consecutive
repeats are suppressed until `repeat_minutes` elapses, when one heartbeat
is re-sent; a code change re-arms immediately and carries a note about
the suppressed run it ended. Net effect: every refusal reason reaches the
phone at least once, without one message per 15m cycle in a 24h chop.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone


class RejectionDedup:
    def __init__(self, repeat_minutes: int) -> None:
        self.repeat_minutes = max(1, int(repeat_minutes))
        self._code: str | None = None
        self._first_at: datetime | None = None
        self._total = 0
        self._last_sent_mono = float("-inf")

    def should_send(
        self, code: str, now: datetime | None = None, mono: float | None = None
    ) -> tuple[bool, str | None]:
        """Decide one rejection occurrence; returns (send, note-or-None)."""
        mono = time.monotonic() if mono is None else mono
        now = now or datetime.now(timezone.utc)
        if code != self._code:
            note = None
            if self._code is not None and self._total > 1:
                t = self._first_at.strftime("%H:%M UTC") if self._first_at else "?"
                note = f"Code précédent {self._code} ×{self._total} (dernier refus à {t})"
            self._code = code
            self._first_at = now
            self._total = 1
            self._last_sent_mono = mono
            return True, note
        self._total += 1
        if mono - self._last_sent_mono >= self.repeat_minutes * 60:
            self._last_sent_mono = mono
            t = self._first_at.strftime("%H:%M UTC") if self._first_at else "?"
            return True, f"Même refus depuis {t} — {self._total} cycles consécutifs"
        return False, None
