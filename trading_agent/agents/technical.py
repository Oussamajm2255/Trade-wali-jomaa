"""Technical-analysis agent (spec §12): DeepSeek over the canonical
snapshot — indicators, HTF bias, MTF biases, 15m structure, liquidity,
DXY context, session context, gold context. The AI only interprets
pre-computed numbers; it never fetches or invents market data."""
from __future__ import annotations

import json
import logging

from trading_agent.agents.base import LLMClient
from trading_agent.agents.fallback import technical_fallback
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, TechnicalOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the TECHNICAL agent inside a gold (XAUUSD) trading decision pipeline.
You receive ONE structured snapshot of precomputed, deterministic market data for the
entry timeframe plus multi-timeframe context. You are an interpreter, not a calculator.

Rules:
1. Base every conclusion ONLY on the numbers provided. You must NEVER claim:
   news you were not given, economic events you were not given, prices you were
   not given, indicators you were not given, or structure you were not given.
2. Use the provided keys: indicators (RSI/ADX/ATR/EMAs/MACD/Bollinger),
   "htf_bias" (4h directional bias), "mtf_biases", "alignment", "regime" and
   "htf_regimes" (deterministic regime engine), "structure" (swings, BOS/CHoCH,
   equal highs/lows, sweeps, FVGs, order blocks, support/resistance),
   "gold_context", "dxy_context" and "session_context".
3. Respond with a single JSON object with exactly these keys:
   {"bias": "long"|"short"|"neutral",
    "conviction": <float 0.0-1.0>,
    "setup_type": "<short setup label, or '' if no setup>",
    "structure_alignment": <float -1.0..1.0, how well market structure agrees
                            with the bias>,
    "support": <number|null>,
    "resistance": <number|null>,
    "reasoning": "2-3 sentences of reasoning from the provided data",
    "invalidating_condition": "what would prove this bias wrong"}
If bias is "neutral", conviction must be <= 0.3.
Support/resistance must come from the provided structure/Bollinger levels only."""


class TechnicalAgent:
    name = "technical"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def analyze(self, snapshot: dict) -> AgentVerdict:
        if self.client.enabled:
            user = json.dumps(snapshot, indent=2)
            output, failure = self.client.complete_model(SYSTEM_PROMPT, user, TechnicalOutput)
            if output is not None:
                return AgentVerdict(
                    agent=self.name,
                    model=self.settings.deepseek_model,
                    payload=output.model_dump(),
                    source="llm",
                )
        else:
            failure = None
        output = technical_fallback(snapshot)
        return AgentVerdict(
            agent=self.name,
            model="heuristic-fallback",
            payload=output.model_dump(),
            source="fallback",
            failure_reason=failure,
        )
