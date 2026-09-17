"""Feature contribution (spec §47): WHAT SUPPORTS / CONTRADICTS /
INVALIDATES a signal — derived from actual calculated values only.

Every factor here maps to a stored, deterministic value (setup-quality
components, conflict report, alignment label, structure features, DXY
classification, SL). Nothing is invented; the LLM has no input here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Setup-quality components (spec §18) with human-readable labels.
_COMPONENT_LABELS = {
    "mtf": "alignement MTF",
    "mtf_alignment": "alignement MTF",
    "structure": "structure de marché",
    "regime": "régime",
    "dxy": "contexte DXY",
    "volatility": "volatilité",
    "session": "session",
    "risk_reward": "ratio risque/rendement",
    "location": "emplacement (liquidité/VWAP)",
}

_SUPPORT_MIN = 0.6
_CONTRADICT_MAX = 0.4


@dataclass
class FeatureContribution:
    """The three §47 lists, traceable to the record that produced them."""

    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    invalidation: str = ""

    def to_dict(self) -> dict:
        return {
            "supporting": self.supporting,
            "contradicting": self.contradicting,
            "invalidation": self.invalidation,
        }


def feature_contribution(record: dict) -> FeatureContribution:
    """Build the §47 explanation from one signal record's stored JSON.

    `record` is a SignalRecord dict (spec §22): market_snapshot,
    setup_quality, conflicts, sl. Everything below is read-only.
    """
    out = FeatureContribution()
    snap = record.get("market_snapshot") or {}
    quality = record.get("setup_quality") or {}
    components = quality.get("components") or {}

    # --- supporting: deterministic context agreeing with the trade ------
    for key, score in sorted(components.items()):
        if isinstance(score, (int, float)) and score >= _SUPPORT_MIN:
            label = _COMPONENT_LABELS.get(key, key)
            out.supporting.append(f"{label} fort ({score:.2f})")
    alignment = ((snap.get("alignment") or {}).get("label")) or ""
    if alignment:
        out.supporting.append(f"alignement {alignment}")
    structure = snap.get("structure") or {}
    present = [
        label for key, label in (("bos", "BOS"), ("choch", "CHoCH"),
                                 ("fvgs", "FVG"), ("sweeps", "balayage de liquidité"))
        if structure.get(key)
    ]
    if present:
        out.supporting.append("structure: " + ", ".join(present))
    dxy = ((snap.get("dxy_context") or snap.get("dxy_gauge")) or {}).get("classification")
    if dxy:
        out.supporting.append(f"DXY: {dxy}")

    # --- contradicting: weak components + deterministic conflicts ------
    for key, score in sorted(components.items()):
        if isinstance(score, (int, float)) and score <= _CONTRADICT_MAX:
            label = _COMPONENT_LABELS.get(key, key)
            out.contradicting.append(f"{label} faible ({score:.2f})")
    conflicts = (record.get("conflicts") or {}).get("conflicts") or []
    for conflict in conflicts:
        detail = conflict.get("detail") if isinstance(conflict, dict) else str(conflict)
        if detail:
            out.contradicting.append(detail)

    # --- invalidation: the stop, the one concrete line ------------------
    sl = record.get("sl")
    if sl is not None:
        out.invalidation = f"setup invalidé si le prix clôture au-delà du SL ({sl:,.8g})"
    elif out.contradicting:
        out.invalidation = "pas de niveau calculé — rejeté avant sizing"
    return out
