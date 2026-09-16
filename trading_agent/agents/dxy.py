"""DXY context agent (spec §14) — internally named DXY_CONTEXT_AGENT.

Interprets the deterministic DXY context (level, direction, momentum,
trend, changes) plus the computed XAUUSD response/divergence and turns
it into a gold bias. This is NOT a news-sentiment agent: the robot has
no news source, and this module never pretends otherwise."""
from __future__ import annotations

import json
import logging

from trading_agent.agents.base import LLMClient
from trading_agent.agents.fallback import dxy_fallback
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, DxyOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the DXY_CONTEXT_AGENT inside a gold (XAUUSD) trading decision pipeline.
You receive a deterministic DXY context block (level, direction, momentum, trend,
15m/1h/4h changes, 0-100 score where 100 = weak dollar = bullish gold) plus the
computed XAUUSD response/divergence. You interpret; you never calculate or invent.

Rules:
1. Base every conclusion ONLY on the provided numbers. This is NOT a news-sentiment
   agent — the robot has no news source, so never mention news or events.
2. Respond with a single JSON object with exactly these keys:
   {"gold_bias": "long"|"short"|"neutral",
    "score": <float -1.0..1.0, signed gold-bullish strength>,
    "dxy_state": "<short label of the dollar state>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "2-3 sentences referencing the provided numbers"}
3. The score sign must agree with gold_bias: long -> positive score, short ->
   negative score, neutral -> score 0.0.
4. A clear divergence (gold and DXY moving in the same direction) must be stated
   in your reasoning and should lower confidence."""


class DxyContextAgent:
    name = "dxy"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def analyze(self, snapshot: dict, gauge: dict | None) -> AgentVerdict:
        if self.client.enabled:
            context = {
                "dxy_context": snapshot.get("dxy_context"),
                "dxy_gauge": gauge,
                "gold_context": {
                    "price": snapshot.get("last_close"),
                    "distance_from_daily_open_pct": (snapshot.get("gold_context") or {}).get(
                        "distance_from_daily_open_pct"
                    ),
                },
            }
            user = json.dumps(context, indent=2)
            output, failure = self.client.complete_model(SYSTEM_PROMPT, user, DxyOutput)
            if output is not None:
                return AgentVerdict(
                    agent=self.name,
                    model=self.settings.deepseek_model,
                    payload=output.model_dump(),
                    source="llm",
                )
        else:
            failure = None
        output = dxy_fallback(snapshot.get("dxy_context"), gauge)
        return AgentVerdict(
            agent=self.name,
            model="heuristic-fallback",
            payload=output.model_dump(),
            source="fallback",
            failure_reason=failure,
        )
