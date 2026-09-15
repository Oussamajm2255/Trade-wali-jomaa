"""Multi-timeframe alignment (spec §5).

Combines the per-timeframe deterministic biases into one classification:
BULLISH_ALIGNMENT / BEARISH_ALIGNMENT / MIXED / CONFLICTED. The robot
must recognise when lower timeframes conflict with the higher ones —
that signal is used later by the setup-quality engine, and is never
computed by the LLM.
"""
from __future__ import annotations

ALIGNMENT_ORDER = ["1d", "4h", "1h", "15m"]


def classify_alignment(biases: dict[str, dict]) -> dict:
    """Score how aligned the timeframes are (-1 bearish .. +1 bullish).

    Rules (deterministic):
    - at least 2 directional timeframes in the same direction and none
      opposing -> BULLISH_ALIGNMENT / BEARISH_ALIGNMENT
    - bullish AND bearish timeframes -> CONFLICTED (the robot must see it)
    - 0-1 directional timeframes, or nothing decisive -> MIXED
    """
    bulls = 0
    bears = 0
    for tf in ALIGNMENT_ORDER:
        bias = (biases.get(tf) or {}).get("bias")
        if bias == "bull":
            bulls += 1
        elif bias == "bear":
            bears += 1
    total = bulls + bears
    if total == 0:
        return {
            "alignment": "MIXED",
            "alignment_score": 0.0,
            "bull_tfs": 0,
            "bear_tfs": 0,
            "detail": "no directional bias on any timeframe (chop everywhere)",
        }
    score = round((bulls - bears) / total, 4)
    if bulls and bears:
        alignment = "CONFLICTED"
        detail = f"{bulls} bullish vs {bears} bearish timeframe(s)"
    elif bulls >= 2 and score > 0:
        alignment = "BULLISH_ALIGNMENT"
        detail = f"{bulls} bullish timeframe(s), no opposing bias"
    elif bears >= 2 and score < 0:
        alignment = "BEARISH_ALIGNMENT"
        detail = f"{bears} bearish timeframe(s), no opposing bias"
    else:
        alignment = "MIXED"
        detail = f"only {total} directional timeframe(s) — not enough alignment"
    return {
        "alignment": alignment,
        "alignment_score": score,
        "bull_tfs": bulls,
        "bear_tfs": bears,
        "detail": detail,
    }
