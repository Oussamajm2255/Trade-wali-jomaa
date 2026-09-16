"""Market-regime agent (spec §13): the AI INTERPRETS the deterministic
regime engine's output (per-timeframe regimes, ADX, ATR behaviour, EMA
alignment/slope) — it never calculates or invents market data."""
from __future__ import annotations

import json
import logging

from trading_agent.agents.base import LLMClient
from trading_agent.agents.fallback import regime_fallback
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, RegimeOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the REGIME agent inside a gold (XAUUSD) trading decision pipeline.
You receive the output of a DETERMINISTIC regime engine (computed locally, not by you)
plus multi-timeframe context. Your job is to interpret it, not to recalculate it.

Rules:
1. Never calculate or invent market data — use only the provided numbers.
2. "regime" (deterministic, entry timeframe) and "htf_regimes" are authoritative:
   trend_up / trend_down / range / high_volatility / low_volatility / transition.
3. Respond with a single JSON object with exactly these keys:
   {"regime": "trending_up"|"trending_down"|"ranging"|"high_volatility",
    "trend_direction": "up"|"down"|"flat",
    "trend_strength": <float 0.0-1.0>,
    "volatility_state": "expanded"|"contracted"|"normal",
    "confidence": <float 0.0-1.0, how confident you are in the interpretation>,
    "reasoning": "2-3 sentences referencing the provided numbers"}
Map high_volatility/low_volatility/range/transition regimes to trend_direction "flat"
unless the provided numbers clearly justify a direction."""


class RegimeAgent:
    name = "regime"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def analyze(self, snapshot: dict) -> AgentVerdict:
        if self.client.enabled:
            structure = snapshot.get("structure") or {}
            context = {
                "deterministic_regime": snapshot.get("regime"),
                "htf_regimes": snapshot.get("htf_regimes"),
                "adx_14": snapshot.get("adx_14"),
                "atr_14": snapshot.get("atr_14"),
                "atr_percentile_100": snapshot.get("atr_percentile_100"),
                "ema_alignment": (
                    "bull"
                    if snapshot.get("ema20_gt_ema50") and snapshot.get("ema50_gt_ema200")
                    else "bear"
                    if (not snapshot.get("ema20_gt_ema50")) and (not snapshot.get("ema50_gt_ema200"))
                    else "mixed"
                ),
                "ema20_slope": (snapshot.get("regime") or {}).get("ema20_slope"),
                "structure": {
                    "trend": structure.get("trend"),
                    "swing_count": len(structure.get("swings", [])),
                    "bos_count": len(structure.get("bos", [])),
                    "choch_count": len(structure.get("choch", [])),
                    "support": structure.get("support", []),
                    "resistance": structure.get("resistance", []),
                },
                "volatility": {
                    "atr_14": snapshot.get("atr_14"),
                    "atr_percentile_100": snapshot.get("atr_percentile_100"),
                },
                "last_close": snapshot.get("last_close"),
            }
            user = json.dumps(context, indent=2)
            output, failure = self.client.complete_model(SYSTEM_PROMPT, user, RegimeOutput)
            if output is not None:
                return AgentVerdict(
                    agent=self.name,
                    model=self.settings.deepseek_model,
                    payload=output.model_dump(),
                    source="llm",
                )
        else:
            failure = None
        output = regime_fallback(snapshot)
        return AgentVerdict(
            agent=self.name,
            model="heuristic-fallback",
            payload=output.model_dump(),
            source="fallback",
            failure_reason=failure,
        )
