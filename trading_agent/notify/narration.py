"""Narrative presentation layer (V-MONSTER §38 extension).

A deterministic French narration generated AFTER fusion/engine.py and
risk/engine.py, from the stored signal record ONLY. It re-tells the
already-taken deterministic decision the way a trader would write it —
never with new data, never with an LLM, never changing the verdict.

Guardrails (spec §7 honesty, non-negotiable):
- The tone is capped by the real confidence score: a raw 0.12 reads as
  genuine uncertainty, never as professional assurance.
- While the calibration stays UNVALIDATED (`calibrated_confidence` is
  None) the narration says so — the scores stay indicative.
- Every number and every level in the text comes from the record
  (liquidity, VWAP, stop, target). No level is ever hallucinated.
- The narration is a pure function of (record, side): same inputs,
  same text. Nothing here influences the decision pipeline.

This module never calls the decision engines; it only formats.
"""
from __future__ import annotations

from typing import Any

# Tone ladder, aligned with the raw-confidence floors of the risk engine
# (min_confidence = 0.55) and the Phase H A+/HIGH statistical tier.
_LOW_MAX = 0.55
_HIGH_MIN = 0.75

# Narrative nouns per setup-quality component (deterministic axis name).
_COMPONENT_NOUNS = {
    "mtf": "l'alignement MTF",
    "mtf_alignment": "l'alignement MTF",
    "structure": "la structure",
    "regime": "le régime",
    "dxy": "le contexte DXY",
    "volatility": "la volatilité",
    "session": "la session",
    "risk_reward": "le ratio risque/rendement",
    "location": "l'emplacement",
}

_SIDE_TEXT = {"LONG": "l'achat", "SHORT": "la vente"}

_REGIME_NARR = {
    "trend_up": "la tendance est haussière",
    "trend_down": "la tendance est baissière",
    "high_volatility": "la volatilité est élevée",
    "low_volatility": "la volatilité est basse",
    "range": "le marché est en range",
    "transition": "le marché est en transition",
}

_VWAP_BASIS = {
    "tick": "volume tick",
    "proxy": "volume indicatif",
    "real": "volume réel",
}

# One honest phrase per NoTradeReason member (fusion/types.py).
_REASON_PHRASES = {
    "LOW_CONFIDENCE": "les votes sont trop faibles ou trop partagés pour engager du risque",
    "LOW_SETUP_QUALITY": "la qualité du setup est insuffisante pour le risque payé",
    "MTF_CONFLICT": "les unités de temps se contredisent",
    "REGIME_CONFLICT": "le régime de marché ne correspond pas à la direction du signal",
    "DXY_CONFLICT": "le dollar travaille contre la direction or",
    "STRUCTURE_CONFLICT": "la structure du marché contredit l'entrée",
    "HIGH_VOLATILITY": "la volatilité est trop élevée pour un stop raisonnable",
    "BAD_SPREAD": "le spread est trop large — les frais mangeraient l'avantage",
    "ABNORMAL_SPEED": "le marché bouge anormalement vite",
    "INSUFFICIENT_ROOM": "pas assez de place jusqu'au prochain obstacle pour un ratio honnête",
    "OPPORTUNITY_ACTIVE": "une opportunité est déjà active sur cette configuration",
    "SIGNAL_DUPLICATE": "ce signal a déjà été traité pour cette opportunité",
    "SIGNAL_INVALIDATED": "le signal s'est invalidé entre l'analyse et l'envoi",
    "TOO_LATE": "le délai restant est inférieur à la fenêtre de réaction humaine",
    "LOW_TIER": "le palier statistique de ce signal est sous le plancher exigé",
    "NEWS_RISK": "une fenêtre de news à fort impact est en cours",
    "SHOCK": "le marché subit un choc — les conditions sont anormales",
    "INSUFFICIENT_DATA": "les données ne suffisent pas pour décider",
    "STATISTICAL_EDGE_UNKNOWN": "l'avantage statistique de cette configuration n'est pas démontré",
    "STATISTICAL_QUALITY": "l'échantillon statistique est trop faible pour valider ce setup",
    "RISK_LIMIT": "le risque total dépasse la limite configurée",
}

# Legacy codes found in old records map onto the current reasons.
_LEGACY_ALIASES = {"DXY_FILTER": "DXY_CONFLICT"}

# What would make the robot reconsider, per rejection reason.
_WATCH_CONDITIONS = {
    "LOW_CONFIDENCE": "un signal plus net émerge (BOS ou CHoCH confirmé)",
    "LOW_SETUP_QUALITY": "la qualité du setup se renforce",
    "MTF_CONFLICT": "les unités de temps se réalignent dans le même sens",
    "REGIME_CONFLICT": "le régime redevient compatible avec la direction",
    "DXY_CONFLICT": "le contexte dollar s'aligne à nouveau avec l'or",
    "STRUCTURE_CONFLICT": "la structure confirme une cassure nette dans le sens du signal",
    "HIGH_VOLATILITY": "la volatilité revient dans une fourchette normale",
    "BAD_SPREAD": "le spread se resserre",
    "ABNORMAL_SPEED": "la vitesse de marché revient à la normale",
    "INSUFFICIENT_ROOM": "un meilleur ratio se présente (objectif plus loin ou stop plus serré)",
    "OPPORTUNITY_ACTIVE": "l'opportunité en cours se résolve (clôture ou invalidation)",
    "SIGNAL_DUPLICATE": "une nouvelle configuration apparaisse",
    "SIGNAL_INVALIDATED": "le prix se stabilise dans la zone visée",
    "TOO_LATE": "un nouveau trigger se forme plus tôt dans la fenêtre",
    "LOW_TIER": "l'historique statistique du bucket remonte au-dessus du plancher",
    "NEWS_RISK": "la fenêtre de news se referme",
    "SHOCK": "le marché se stabilise",
    "INSUFFICIENT_DATA": "assez de données s'accumulent pour décider",
    "STATISTICAL_EDGE_UNKNOWN": "l'historique démontre un avantage mesurable",
    "STATISTICAL_QUALITY": "l'échantillon s'étoffe suffisamment",
    "RISK_LIMIT": "l'exposition actuelle redescend sous la limite",
}


def tone_tier(fusion: dict | None) -> str | None:
    """LOW/MEDIUM/HIGH from the stored fusion context.

    The Phase H tier (fusion.tier) wins when present — it is the
    statistical verdict. Otherwise the raw confidence is banded with
    the risk engine's own floors: < 0.55 LOW (the LOW_CONFIDENCE gate),
    >= 0.75 HIGH, in between MEDIUM. No confidence data -> None (the
    narration then avoids tone words entirely).
    """
    fusion = fusion or {}
    tier = fusion.get("tier")
    if tier in ("LOW", "MEDIUM", "HIGH"):
        return str(tier)
    raw = fusion.get("raw_confidence")
    if not isinstance(raw, (int, float)):
        return None
    raw = float(raw)
    if raw < _LOW_MAX:
        return "LOW"
    if raw >= _HIGH_MIN:
        return "HIGH"
    return "MEDIUM"


def build_narrative(record: dict, side: str | None = None, rejected: bool | None = None) -> str:
    """The narrative block for one already-taken decision.

    Pure function of the record: same inputs, same text. Returns ""
    only when the record carries nothing to tell (never happens for a
    stored decision, kept as a safety net).
    """
    snap = record.get("market_snapshot") or {}
    fusion = record.get("fusion") or {}
    if rejected is None:
        rejected = bool(record.get("no_trade_reason")) or record.get("final_decision") == "rejected"
    tier = tone_tier(fusion)
    state = _market_state(snap)

    paragraphs: list[str] = []
    if rejected:
        paragraphs.append(_rejection_opener(record, tier, state))
    else:
        paragraphs.append(_proposal_opener(tier, state, side))
    why = _merged_why(record, side)
    if why:
        paragraphs.append(_cap(why))
    paragraphs.append(_watch_line(record, side, rejected))
    hedge = _hedge(fusion)
    if hedge:
        paragraphs.append(hedge)
    return "\n".join(paragraphs)


# ------------------------------------------------------------------ openers


def _proposal_opener(tier: str | None, state: str, side: str | None) -> str:
    """The 1-2 sentence opening: what is happening + the decision,
    worded no stronger than the real confidence."""
    side_text = _SIDE_TEXT.get(side or "", "une entrée")
    if tier == "LOW":
        base = "Je n'ai pas de conviction claire" + (f" — {state}" if state else "")
        return f"{base}. Je propose {side_text} avec prudence."
    if tier == "HIGH":
        prefix = f"{_cap(state)} — le contexte est net" if state else "Le contexte est net"
        return f"{prefix}. Je propose {side_text} avec conviction."
    # MEDIUM or uncalibrated: measured, exploitable but not perfect.
    if state:
        return (
            f"{_cap(state)} — un contexte exploitable sans être parfait. "
            f"Je propose {side_text} avec une conviction mesurée."
        )
    return (
        "Le contexte est exploitable sans être parfait. "
        f"Je propose {side_text} avec une conviction mesurée."
    )


def _rejection_opener(record: dict, tier: str | None, state: str) -> str:
    """Why we stay out — the refusal reason told as a sentence."""
    code = record.get("no_trade_reason")
    key = _LEGACY_ALIASES.get(code, code)
    phrase = _REASON_PHRASES.get(key, "les conditions ne sont pas réunies")
    conflicts = record.get("conflicts") or {}
    if key == "LOW_CONFIDENCE":
        opener = f"Je n'ai pas de conviction claire — {phrase}. Je reste à l'écart."
    elif conflicts.get("state") == "CONFLICTED":
        if "contredisent" in phrase:
            opener = f"{_cap(phrase)}. Je reste à l'écart."
        else:
            opener = f"Les signaux se contredisent nettement — {phrase}. Je reste à l'écart."
    elif tier == "LOW":
        opener = f"La conviction était de toute façon limitée — {phrase}. Je reste à l'écart."
    elif tier == "HIGH":
        opener = f"Le setup avait du potentiel, mais {phrase}. Je reste à l'écart — la discipline prime."
    elif tier == "MEDIUM":
        opener = f"La direction était lisible, mais {phrase}. Je reste à l'écart."
    else:
        opener = f"{_cap(phrase)}. Je reste à l'écart."
    if state:
        opener = f"{_cap(state)}. {opener}"
    return opener


# ------------------------------------------------------------ market state


def _market_state(snap: dict) -> str:
    """What is happening on the market, from stored snapshot facts only."""
    clauses: list[str] = []
    structure = snap.get("structure") or {}
    liq = snap.get("liquidity") or {}
    if structure.get("sweeps"):
        below = (liq.get("nearest_below") or {}).get("price")
        above = (liq.get("nearest_above") or {}).get("price")
        if below is not None:
            clauses.append(f"le marché vient de balayer les liquidités sous {_fmt(below)}")
        elif above is not None:
            clauses.append(f"le marché vient de balayer les liquidités au-dessus de {_fmt(above)}")
        else:
            clauses.append("un balayage de liquidité vient de se produire")
    elif structure.get("bos"):
        clauses.append("une cassure de structure (BOS) est en place")
    elif structure.get("choch"):
        clauses.append("un retournement de structure (CHoCH) s'est formé")
    regime = (snap.get("regime") or {}).get("regime")
    if regime in _REGIME_NARR:
        clauses.append(_REGIME_NARR[regime])
    dxy = ((snap.get("dxy_context") or snap.get("dxy_gauge")) or {}).get("classification") or ""
    dxy_l = str(dxy).lower()
    if "weak" in dxy_l:
        clauses.append("le dollar est faible")
    elif "strong" in dxy_l:
        clauses.append("le dollar est fort")
    session = (snap.get("session_context") or {}).get("session")
    if session == "LONDON":
        clauses.append("en pleine session de Londres")
    elif session == "NEW_YORK":
        clauses.append("en session de New York")
    return ", ".join(clauses[:3])


# ------------------------------------------------------------------- why


def _tension_nouns(record: dict) -> tuple[list[str], list[str]]:
    """The real tension between signals, as nouns — not two lists."""
    sup: list[str] = []
    con: list[str] = []
    quality = (record.get("setup_quality") or {}).get("components") or {}
    for key, score in quality.items():
        noun = _COMPONENT_NOUNS.get(key)
        if not noun or not isinstance(score, (int, float)):
            continue
        if score >= 0.6:
            sup.append(noun)
        elif score <= 0.4:
            con.append(noun)
    alignment = ((record.get("market_snapshot") or {}).get("alignment") or {}).get("label")
    if alignment == "aligned":
        sup.append("l'alignement")
    conflicts = (record.get("conflicts") or {}).get("conflicts") or []
    if conflicts:
        con.append("les conflits non résolus")
    return list(dict.fromkeys(sup))[:3], list(dict.fromkeys(con))[:2]


def _merged_why(record: dict, side: str | None) -> str:
    """The WHY told as one logic — the tension, not an enumeration."""
    noun_side = _SIDE_TEXT.get(side or "", "une entrée")
    sup, con = _tension_nouns(record)
    if not sup and not con:
        return ""
    if sup and con:
        s_verb = "penchent" if len(sup) > 1 else "penche"
        c_verb = "retiennent" if len(con) > 1 else "retient"
        return (
            f"{_join(sup)} {s_verb} pour {noun_side}, mais {_join(con)} "
            f"{c_verb} le déclencheur — la tension est réelle."
        )
    if sup:
        return f"Presque tout converge ici : {_join(sup)} penchent pour {noun_side}."
    return f"Les freins dominent : {_join(con)}."


# ------------------------------------------------------------------ watch


def _watch_line(record: dict, side: str | None, rejected: bool) -> str:
    """The concrete next trigger, from stored liquidity/VWAP/stop only."""
    snap = record.get("market_snapshot") or {}
    liq = snap.get("liquidity") or {}
    vwap = snap.get("vwap") or {}
    sl = record.get("sl")
    tp = record.get("tp")

    if rejected:
        code = record.get("no_trade_reason")
        key = _LEGACY_ALIASES.get(code, code)
        cond = _WATCH_CONDITIONS.get(key, "les conditions se normalisent")
        target = (liq.get("nearest_below") or {}) or (liq.get("nearest_above") or {})
        price = target.get("price")
        if price is not None:
            kind = target.get("kind") or "liquidité"
            return f"À surveiller : {cond}. Je surveille {_fmt(price)} ({kind}) comme prochain déclencheur."
        return f"À surveiller : {cond}. Aucun niveau calculé — observation du prix uniquement."

    parts: list[str] = []
    if side == "LONG":
        above = liq.get("nearest_above") or {}
        if above.get("price") is not None:
            parts.append(
                f"la liquidité au-dessus à {_fmt(above['price'])} "
                f"({above.get('kind') or 'liquidité'}) comme prochain objectif"
            )
        elif tp is not None:
            parts.append(f"l'objectif à {_fmt(tp)}")
    elif side == "SHORT":
        below = liq.get("nearest_below") or {}
        if below.get("price") is not None:
            parts.append(
                f"la liquidité en dessous à {_fmt(below['price'])} "
                f"({below.get('kind') or 'liquidité'}) comme prochain objectif"
            )
        elif tp is not None:
            parts.append(f"l'objectif à {_fmt(tp)}")
    if sl is not None:
        parts.append(f"{_fmt(sl)} invalide le setup")
    if not parts and vwap.get("available") and vwap.get("session_vwap") is not None:
        basis = _VWAP_BASIS.get(str(vwap.get("volume_basis")), "")
        marker = f" ({basis})" if basis else ""
        parts.append(f"le VWAP session à {_fmt(vwap['session_vwap'])}{marker} comme référence")
    if not parts:
        return "Aucun niveau calculé pour l'instant — j'observe le prix."
    return "À surveiller : " + " ; ".join(parts) + "."


# ------------------------------------------------------------------ hedge


def _hedge(fusion: dict) -> str:
    """The UNVALIDATED hedge: honest about a score that is not yet a
    calibrated probability (§7). Shown only when a raw score exists."""
    raw = fusion.get("raw_confidence")
    if isinstance(raw, (int, float)) and fusion.get("calibrated_confidence") is None:
        return (
            "Ma calibration n'est pas encore validée statistiquement — "
            "ces scores restent indicatifs, pas des probabilités."
        )
    return ""


# ----------------------------------------------------------------- helpers


def _fmt(value: Any) -> str:
    return f"{float(value):,.2f}"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} et {items[1]}"
    return ", ".join(items[:-1]) + f" et {items[-1]}"
