"""Telegram notifications — real-time signals on the user's phone.

One-way for now: the robot pushes startup confirmations, trade signals and
kill-switch alerts. Decisions stay in the CLI (`approve` / `reject`) — the
robot never executes on its own. Every send is best-effort: a failed
notification must never break the trading loop.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from trading_agent.config import Settings
from trading_agent.schema.types import SignalProposal

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self.s = settings

    @property
    def enabled(self) -> bool:
        return bool(self.s.telegram_bot_token and self.s.telegram_chat_id)

    def send(self, text: str) -> bool:
        """Send a message; best-effort, never raises."""
        if not self.enabled:
            return False
        try:
            import httpx

            url = f"{API_BASE}/bot{self.s.telegram_bot_token}/sendMessage"
            resp = httpx.post(url, json={"chat_id": self.s.telegram_chat_id, "text": text}, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as exc:  # noqa: BLE001 - network failures are non-fatal
            logger.warning("telegram send failed: %s", exc)
            return False

    # ------------------------------------------------------------- templates

    def send_startup(self, symbols: list[str], timeframe: str, mode: str) -> bool:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        text = (
            "✅ Robot de trading EN LIGNE\n"
            "Vous êtes dans la boucle.\n\n"
            f"Symboles : {', '.join(symbols)}\n"
            f"Unité de temps : {timeframe}\n"
            f"Mode : {mode}\n"
            f"Filtre DXY : {'actif' if self.s.dxy_filter_enabled else 'désactivé'}\n\n"
            f"{ts}"
        )
        return self.send(text)

    def send_signal(self, proposal: SignalProposal, gauge: dict | None) -> bool:
        dxy_line = ""
        if gauge and gauge.get("kind") == "dxy":
            dxy_line = f"DXY : {gauge.get('value')} ({gauge.get('classification')})\n"
        text = (
            f"🎯 SIGNAL {proposal.symbol} — {proposal.side.value.upper()}\n\n"
            f"Entrée : {proposal.entry:,.2f}\n"
            f"Stop : {proposal.stop:,.2f}\n"
            f"Objectif : {proposal.target:,.2f}\n"
            f"Taille : {proposal.size:,.4f}\n"
            f"Risque : {proposal.risk_amount:,.2f} $ | Confiance : {proposal.confidence:.2f}\n"
            f"{dxy_line}\n"
            f"✅ Approuver : python -m trading_agent.main approve {proposal.id}\n"
            f"❌ Rejeter  : python -m trading_agent.main reject {proposal.id}"
        )
        return self.send(text)

    def send_alert(self, text: str) -> bool:
        return self.send(f"🚨 {text}")

    def send_test(self) -> bool:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return self.send(f"✅ Signal de test — vous êtes dans la boucle.\n{ts}")
