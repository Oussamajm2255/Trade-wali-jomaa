"""Telegram notifications — real-time signals on the user's phone.

One-way for now: the robot pushes startup confirmations, full spec §38
trade signals and kill-switch alerts; rejected opportunities are sent
only when `telegram_rejection_alerts` is on. Decisions stay in the CLI
(`approve` / `reject`) — the robot never executes on its own. Every
send is best-effort: a failed notification must never break the loop.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from trading_agent.analytics.contribution import feature_contribution
from trading_agent.config import Settings
from trading_agent.schema.types import SignalProposal

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"

# Canonical MTF display order for the bias line (spec §38).
_MTF_ORDER = ["1d", "4h", "1h", "15m"]
_TF_LABELS = {"1d": "1D", "4h": "4H", "1h": "1H", "15m": "15m", "5m": "5m", "30m": "30m"}
_BIAS_LABELS = {"long": "haussière", "short": "baissière", "neutral": "neutre"}
_DXY_TREND_LABELS = {"bull": "haussière", "bear": "baissière", "flat": "plate"}
_REGIME_LABELS = {
    "trend_up": "tendance haussière",
    "trend_down": "tendance baissière",
    "high_volatility": "volatilité élevée",
    "low_volatility": "volatilité basse",
    "transition": "transition",
    "range": "range",
}
_SESSION_LABELS = {
    "ASIA": "Asie",
    "SYDNEY": "Sydney",
    "LONDON": "Londres",
    "NEW_YORK": "New York",
    "LONDON_NY_OVERLAP": "Chevauchement Londres+NY",
    "OFF_SESSION": "hors session",
}
_VWAP_STATE_LABELS = {
    "reclaimed": "réclamation du VWAP",
    "rejected": "rejet du VWAP",
    "above": "au-dessus du VWAP",
    "below": "sous le VWAP",
}
_SPEED_LABELS = {
    "SLOW": "Lente",
    "NORMAL": "Normale",
    "FAST": "Rapide",
    "EXTREME": "Extrême",
}


def agent_bias(payload: dict) -> str | None:
    """Each agent names its direction differently: technical uses
    'bias', dxy uses 'gold_bias', regime uses 'trend_direction'/'regime'."""
    for key in ("bias", "side", "gold_bias", "trend_direction", "regime"):
        value = payload.get(key)
        if value:
            return str(value)
    return None


def agent_conviction(payload: dict) -> float | None:
    """Confidence under each agent's own key name ('conviction' or
    'confidence'); 'score' is excluded — it is a signed direction score
    (dxy), not a 0-1 conviction."""
    for key in ("conviction", "confidence"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


class TelegramNotifier:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        # Phase A (V-MONSTER §5): last send round-trip in ms (0 = never
        # sent) — logged per cycle and available for health checks.
        self.last_latency_ms = 0.0
        # Phase K (§76): delivery counters for the health score.
        self.sends = 0
        self.failures = 0

    @property
    def enabled(self) -> bool:
        return bool(self.s.telegram_bot_token and self.s.telegram_chat_id)

    def send(self, text: str) -> bool:
        """Send a message; best-effort, never raises."""
        if not self.enabled:
            return False
        start = time.monotonic()
        try:
            import httpx

            url = f"{API_BASE}/bot{self.s.telegram_bot_token}/sendMessage"
            resp = httpx.post(url, json={"chat_id": self.s.telegram_chat_id, "text": text}, timeout=10)
            resp.raise_for_status()
            self.last_latency_ms = (time.monotonic() - start) * 1000
            self.sends += 1
            return True
        except Exception as exc:  # noqa: BLE001 - network failures are non-fatal
            self.last_latency_ms = (time.monotonic() - start) * 1000
            self.sends += 1
            self.failures += 1
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

    # Phase I (§62): supervision follow-ups, sent only on state CHANGE.
    _SUPERVISION_LABELS = {
        "VALID": "entrée toujours valable",
        "DO_NOT_CHASE": "ne pas chasser l'entrée",
        "INVALIDATED": "signal invalidé",
        "EXPIRED": "signal expiré",
    }

    def supervision_message(self, event: dict) -> str:
        """Concise follow-up when a pending proposal's state changes."""
        state = event.get("state", "?")
        label = self._SUPERVISION_LABELS.get(state, state)
        lines = [
            f"🔄 SUIVI {event.get('symbol', '?')} : {label}",
            f"Détail : {event.get('detail', '')}",
        ]
        previous = event.get("previous_state")
        if previous:
            lines.append(f"État précédent : {previous}")
        lines.append(f"Proposal : {event.get('proposal_id', '?')}")
        return "\n".join(lines)

    def send_supervision(self, event: dict) -> bool:
        """Send a supervision follow-up (state-change only — dedup'd)."""
        return self.send(self.supervision_message(event))

    def proposal_message(
        self,
        proposal: SignalProposal,
        gauge: dict | None = None,
        record: dict | None = None,
        proposal_id: str | None = None,
    ) -> str:
        """Full spec §38 proposal text — every field from the stored record.

        With `record` (the SignalRecord dict, spec §22) the message shows
        the complete traceable field list: MTF biases, regime, DXY,
        session, structure, §47 contribution, risk-gate trail, AI agent
        summaries and strategy version. Without a record it falls back to
        the legacy short format — nothing is invented.
        """
        if record is None:
            return self._legacy_signal_text(proposal, gauge, proposal_id)
        snap = record.get("market_snapshot") or {}
        fusion = record.get("fusion") or {}
        quality = record.get("setup_quality") or {}
        conflicts = record.get("conflicts") or {}
        gates = record.get("gates") or []
        outputs = record.get("ai_outputs") or {}
        contribution = feature_contribution(record)

        lines = [
            f"🎯 SIGNAL {proposal.symbol} — {proposal.side.value.upper()}",
            "",
            f"Entrée : {proposal.entry:,.2f}",
            f"Stop : {proposal.stop:,.2f}",
            f"Objectif : {proposal.target:,.2f}",
            f"RR : {proposal.expected_rr or 0.0:.2f} | Taille : {proposal.size:,.4f}"
            f" | Risque : {proposal.risk_amount:,.2f} $",
            "",
        ]

        calibrated = fusion.get("calibrated_confidence")
        conf = f"Confiance brute : {proposal.confidence:.2f}"
        if calibrated is not None:
            conf += f" | Calibrée : {float(calibrated):.2f}"
        lines.append(conf)
        sq = quality.get("score")
        if sq is not None:
            state = conflicts.get("state")
            lines.append(f"Qualité du setup : {sq:.2f}" + (f" (conflits : {state})" if state else ""))
        # Phase H (§81): A+/A label with its confidence tier.
        label = record.get("signal_label")
        tier = fusion.get("tier")
        if label:
            lines.append(f"Label : {label}" + (f" (palier {tier})" if tier else ""))

        mtf = snap.get("mtf_biases") or {}
        ordered = [tf for tf in _MTF_ORDER if tf in mtf] + sorted(set(mtf) - set(_MTF_ORDER))
        if ordered:
            parts = [
                f"{_TF_LABELS.get(tf, tf)} "
                f"{_BIAS_LABELS.get(str(mtf[tf].get('bias')).lower(), mtf[tf].get('bias'))}"
                for tf in ordered
            ]
            lines.append("Biais MTF : " + " · ".join(parts))
        regime = (snap.get("regime") or {}).get("regime")
        dxy_ctx = snap.get("dxy_context") or {}
        dxy_gauge = snap.get("dxy_gauge") or {}
        dxy_class = dxy_ctx.get("classification") or dxy_gauge.get("classification")
        dxy_trend = dxy_ctx.get("trend")
        session = (snap.get("session_context") or {}).get("session")
        ctx = []
        if regime:
            ctx.append(f"Régime : {_REGIME_LABELS.get(regime, regime)}")
        if dxy_class:
            ctx.append(f"DXY : {dxy_class}")
        if dxy_trend:
            ctx.append(f"Tendance DXY : {_DXY_TREND_LABELS.get(str(dxy_trend).lower(), dxy_trend)}")
        if session:
            ctx.append(f"Session : {_SESSION_LABELS.get(session, session)}")
        if ctx:
            lines.append(" | ".join(ctx))

        structure = snap.get("structure") or {}
        present = [
            label for key, label in (("bos", "BOS"), ("choch", "CHoCH"),
                                     ("fvgs", "FVG"), ("sweeps", "balayage de liquidité"))
            if structure.get(key)
        ]
        if present:
            lines.append("Structure : " + ", ".join(present))
        if contribution.supporting:
            lines.append("Soutient : " + " · ".join(contribution.supporting))
        if contribution.contradicting:
            lines.append("Contredit : " + " · ".join(contribution.contradicting))
        if contribution.invalidation:
            lines.append("Invalidation : " + contribution.invalidation)
        lines.extend(self._liquidity_vwap_lines(snap))
        lines.extend(self._speed_trigger_lines(snap, record))
        lines.extend(self._timing_lines(record))

        trail = []
        for gate in gates:
            mark = "✓" if gate.get("status") == "pass" else "✗"
            trail.append(f"{gate.get('gate')} {mark}")
        if trail:
            lines.append("Gates : " + " | ".join(trail))
        for name, verdict in sorted(outputs.items()):
            verdict = verdict or {}
            payload = verdict.get("payload") or {}
            piece = f"{name.upper()} : {agent_bias(payload) or '?'}"
            conviction = agent_conviction(payload)
            if conviction is not None:
                piece += f" ({conviction:.2f})"
            if verdict.get("source") == "fallback":
                piece += " [heuristique]"
            lines.append(piece)

        lines += [
            "",
            f"Version : {record.get('strategy_version') or proposal.model}",
            "",
            "⏳ EN ATTENTE D'APPROBATION HUMAINE",
        ]
        if proposal_id:
            lines += [
                f"✅ Approuver : python -m trading_agent.main approve {proposal_id}",
                f"❌ Rejeter  : python -m trading_agent.main reject {proposal_id}",
            ]
        return "\n".join(lines)

    def _legacy_signal_text(
        self, proposal: SignalProposal, gauge: dict | None, proposal_id: str | None
    ) -> str:
        """Pre-record fallback: the short format, kept backward-compatible."""
        dxy_line = ""
        if gauge and gauge.get("kind") == "dxy":
            dxy_line = f"DXY : {gauge.get('value')} ({gauge.get('classification')})\n"
        decision_id = proposal_id or proposal.id
        text = (
            f"🎯 SIGNAL {proposal.symbol} — {proposal.side.value.upper()}\n\n"
            f"Entrée : {proposal.entry:,.2f}\n"
            f"Stop : {proposal.stop:,.2f}\n"
            f"Objectif : {proposal.target:,.2f}\n"
            f"Taille : {proposal.size:,.4f}\n"
            f"Risque : {proposal.risk_amount:,.2f} $ | Confiance : {proposal.confidence:.2f}\n"
            f"{dxy_line}\n"
            f"✅ Approuver : python -m trading_agent.main approve {decision_id}\n"
            f"❌ Rejeter  : python -m trading_agent.main reject {decision_id}"
        )
        return text

    def rejection_message(self, record: dict, note: str | None = None) -> str:
        """Rejected-opportunity alert (§38) with the WHY, traced to the record.

        When the agents ran before the refusal (record has `ai_outputs`)
        the message carries the full analysis: fusion confidence, MTF
        biases, regime/DXY/session context, structure, §47 contributions,
        the gate trail and each agent's verdict + reasoning — the smart
        justification for not putting money in. Pre-AI refusals (data
        down, news blackout, shock) fall back to the concise format;
        nothing is ever invented.
        """
        reason = record.get("decision_reason") or "aucune raison enregistrée"
        code = record.get("no_trade_reason")
        tf = record.get("timeframe")
        lines = [
            f"🚫 SIGNAL REJETÉ {record.get('symbol', '?')}" + (f" ({tf})" if tf else ""),
            f"Raison : {reason}" + (f" ({code})" if code else ""),
        ]
        if note:
            lines.append(note)
        # Phase H (§81): rejections always carry the NO TRADE label.
        label = record.get("signal_label")
        if label:
            lines.append(f"Label : {label}")
        outputs = record.get("ai_outputs") or {}
        if not outputs:
            lines += self._trace_lines(record)
            return "\n".join(lines)

        snap = record.get("market_snapshot") or {}
        fusion = record.get("fusion") or {}
        gates = record.get("gates") or []
        contribution = feature_contribution(record)

        lines.append("")
        raw = fusion.get("raw_confidence")
        if raw is not None:
            conf = f"Confiance brute : {float(raw):.2f}"
            calibrated = fusion.get("calibrated_confidence")
            if calibrated is not None:
                conf += f" | Calibrée : {float(calibrated):.2f}"
            lines.append(conf)

        mtf = snap.get("mtf_biases") or {}
        ordered = [t for t in _MTF_ORDER if t in mtf] + sorted(set(mtf) - set(_MTF_ORDER))
        if ordered:
            parts = [
                f"{_TF_LABELS.get(t, t)} "
                f"{_BIAS_LABELS.get(str(mtf[t].get('bias')).lower(), mtf[t].get('bias'))}"
                for t in ordered
            ]
            lines.append("Biais MTF : " + " · ".join(parts))
        regime = (snap.get("regime") or {}).get("regime")
        dxy_ctx = snap.get("dxy_context") or {}
        dxy_gauge = snap.get("dxy_gauge") or {}
        dxy_class = dxy_ctx.get("classification") or dxy_gauge.get("classification")
        dxy_trend = dxy_ctx.get("trend")
        session = (snap.get("session_context") or {}).get("session")
        ctx = []
        if regime:
            ctx.append(f"Régime : {_REGIME_LABELS.get(regime, regime)}")
        if dxy_class:
            ctx.append(f"DXY : {dxy_class}")
        if dxy_trend:
            ctx.append(f"Tendance DXY : {_DXY_TREND_LABELS.get(str(dxy_trend).lower(), dxy_trend)}")
        if session:
            ctx.append(f"Session : {_SESSION_LABELS.get(session, session)}")
        if ctx:
            lines.append(" | ".join(ctx))

        structure = snap.get("structure") or {}
        present = [
            label
            for key, label in (
                ("bos", "BOS"),
                ("choch", "CHoCH"),
                ("fvgs", "FVG"),
                ("sweeps", "balayage de liquidité"),
            )
            if structure.get(key)
        ]
        if present:
            lines.append("Structure : " + ", ".join(present))
        if contribution.supporting:
            lines.append("Soutient : " + " · ".join(contribution.supporting))
        if contribution.contradicting:
            lines.append("Contredit : " + " · ".join(contribution.contradicting))
        lines.extend(self._liquidity_vwap_lines(snap))
        lines.extend(self._speed_trigger_lines(snap, record))

        trail = []
        for gate in gates:
            mark = "✓" if gate.get("status") == "pass" else "✗"
            trail.append(f"{gate.get('gate')} {mark}")
        if trail:
            lines.append("Gates : " + " | ".join(trail))

        for name, verdict in sorted(outputs.items()):
            verdict = verdict or {}
            payload = verdict.get("payload") or {}
            piece = f"{name.upper()} : {agent_bias(payload) or '?'}"
            conviction = agent_conviction(payload)
            if conviction is not None:
                piece += f" ({conviction:.2f})"
            if verdict.get("source") == "fallback":
                piece += " [heuristique]"
            lines.append(piece)
            reasoning = str(payload.get("reasoning") or "").strip()
            if reasoning:
                snippet = reasoning[:180]
                if len(reasoning) > 180:
                    snippet += "…"
                lines.append(f"  ↳ {snippet}")

        lines += self._trace_lines(record)
        return "\n".join(lines)

    def _liquidity_vwap_lines(self, snap: dict) -> list[str]:
        """Phase B (V-MONSTER §9/§12): VWAP state + nearest liquidity."""
        lines: list[str] = []
        vwap = snap.get("vwap") or {}
        if vwap.get("available"):
            parts = []
            # Honesty (§7): tick-volume VWAP is never presented as traded
            # volume VWAP — the broker basis is labelled on the anchor.
            marker = " (volume tick)" if vwap.get("volume_basis") == "tick" else ""
            if vwap.get("session_vwap") is not None:
                parts.append(f"VWAP session : {float(vwap['session_vwap']):,.2f}{marker}")
            if vwap.get("daily_vwap") is not None:
                parts.append(f"VWAP jour : {float(vwap['daily_vwap']):,.2f}")
            state = vwap.get("state")
            if state:
                parts.append(_VWAP_STATE_LABELS.get(state, state))
            if parts:
                lines.append(" · ".join(parts))
        elif vwap.get("reason"):
            lines.append(f"VWAP : indisponible ({vwap['reason']})")
        liq = snap.get("liquidity") or {}
        above = liq.get("nearest_above")
        below = liq.get("nearest_below")
        if above or below:
            parts = []
            if above:
                parts.append(f"Liquidité ↑ {float(above['price']):,.2f} ({above.get('kind')})")
            if below:
                parts.append(f"Liquidité ↓ {float(below['price']):,.2f} ({below.get('kind')})")
            if liq.get("quality") is not None:
                parts.append(f"qualité {float(liq['quality']):.2f}")
            lines.append(" | ".join(parts))
        return lines

    def _speed_trigger_lines(self, snap: dict, record: dict) -> list[str]:
        """Phase D (V-MONSTER §27/§30): market speed + trigger state."""
        lines: list[str] = []
        speed = snap.get("speed") or {}
        if speed.get("state"):
            suffix = (
                f" ({speed['range_ratio']}x ATR)"
                if speed.get("range_ratio") is not None
                else ""
            )
            lines.append(
                "Vitesse : " + _SPEED_LABELS.get(speed["state"], speed["state"]) + suffix
            )
        trigger = (record.get("fusion") or {}).get("trigger") or {}
        if trigger.get("quality") is not None:
            state = "confirmé" if trigger.get("confirmed") else "non confirmé"
            lines.append(f"Déclencheur : {float(trigger['quality']):.2f} ({state})")
        return lines

    def _timing_lines(self, record: dict) -> list[str]:
        """Phase G (V-MONSTER §42-§49/§64): timing quality, execution
        zone, max chase, expected price/drift, actionability deadline."""
        lines: list[str] = []
        timing = (record.get("fusion") or {}).get("timing") or {}
        if not timing:
            return lines
        quality = timing.get("quality")
        if quality is not None:
            q = float(quality)
            label = (
                "excellent" if q >= 0.8 else "bon" if q >= 0.6
                else "moyen" if q >= 0.4 else "faible"
            )
            lines.append(f"Timing : {q:.2f} ({label})")
        zone = timing.get("execution_zone")
        if zone and len(zone) == 2 and timing.get("max_chase") > 0:
            lines.append(f"Zone d'exécution : {float(zone[0]):,.2f} – {float(zone[1]):,.2f}")
            lines.append(f"Chasse max : {float(timing['max_chase']):,.2f}")
        if timing.get("expected_price") is not None:
            lines.append(
                f"Prix attendu : {float(timing['expected_price']):,.2f}"
                f" (dérive {float(timing['expected_drift'] or 0):,.2f})"
            )
        if timing.get("deadline_iso"):
            lines.append(f"Valable jusqu'à : {timing['deadline_iso']}")
        return lines

    def _trace_lines(self, record: dict) -> list[str]:
        """Data-quality + version footer shared by both rejection formats."""
        lines: list[str] = []
        snap = record.get("market_snapshot") or {}
        quality = snap.get("data_quality")
        if quality:
            lines.append(f"Qualité des données : {quality}")
        version = record.get("strategy_version")
        if version:
            lines.append(f"Version : {version}")
        return lines

    def send_signal(
        self,
        proposal: SignalProposal,
        gauge: dict | None = None,
        record: dict | None = None,
        proposal_id: str | None = None,
    ) -> bool:
        return self.send(self.proposal_message(proposal, gauge, record, proposal_id))

    def send_rejection(self, record: dict, note: str | None = None) -> bool:
        """Rejected-opportunity alert, gated by telegram_rejection_alerts."""
        if not self.s.telegram_rejection_alerts:
            return False
        return self.send(self.rejection_message(record, note))

    def send_alert(self, text: str) -> bool:
        return self.send(f"🚨 {text}")

    def send_test(self) -> bool:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return self.send(f"✅ Signal de test — vous êtes dans la boucle.\n{ts}")
