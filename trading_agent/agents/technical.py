"""Technical-analysis agent: DeepSeek over a precomputed indicator snapshot."""
from __future__ import annotations

import json
import logging

from trading_agent.agents.base import LLMClient
from trading_agent.agents.fallback import technical_fallback
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, TechnicalOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a technical-analysis agent inside a crypto trading decision pipeline.
You receive a structured snapshot of precomputed indicators for one symbol/timeframe.

Rules:
1. Base every conclusion ONLY on the provided numbers. Never invent prices, events or news.
2. Be explicit about disagreement between indicators in your notes.
3. Respond with a single JSON object with exactly these keys:
   {"bias": "long"|"short"|"neutral",
    "conviction": <float 0.0-1.0>,
    "support": <number|null>,
    "resistance": <number|null>,
    "notes": "2-3 sentences of reasoning"}
If bias is "neutral", conviction must be <= 0.3.
Support/resistance must come from the provided Bollinger levels or recent price data only."""


class TechnicalAgent:
    name = "technical"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def analyze(self, snapshot: dict) -> AgentVerdict:
        if self.client.enabled:
            user = json.dumps(snapshot, indent=2)
            output, _ = self.client.complete_model(SYSTEM_PROMPT, user, TechnicalOutput)
            if output is not None:
                return AgentVerdict(
                    agent=self.name,
                    model=self.settings.deepseek_model,
                    payload=output.model_dump(),
                    source="llm",
                )
        output = technical_fallback(snapshot)
        return AgentVerdict(
            agent=self.name,
            model="heuristic-fallback",
            payload=output.model_dump(),
            source="fallback",
        )
