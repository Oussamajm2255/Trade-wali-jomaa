"""Sentiment agent: Fear & Greed index + price/volume context via DeepSeek."""
from __future__ import annotations

import json
import logging

from trading_agent.agents.base import LLMClient
from trading_agent.agents.fallback import sentiment_fallback
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, SentimentOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a market-sentiment agent inside a trading decision pipeline.
You receive a market sentiment gauge (0-100) plus a small price/volume context.
The gauge may come from different sources (crypto Fear & Greed, dollar index
for gold, ...) — the "source" field says which.

Rules:
1. Score must be between -1.0 (extreme bearish) and +1.0 (extreme bullish).
2. Contrarian nuance is allowed but must be stated in "notes".
3. Respond with a single JSON object with exactly these keys:
   {"score": <float -1.0..1.0>, "tone": "<short label>", "notes": "2-3 sentences"}
Never invent news or events."""


class SentimentAgent:
    name = "sentiment"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def analyze(self, gauge: dict | None, snapshot: dict) -> AgentVerdict:
        if self.client.enabled:
            context = {
                "sentiment_gauge": gauge,
                "price_context": {
                    "last_close": snapshot.get("last_close"),
                    "return_24h_pct": snapshot.get("return_24h_pct"),
                    "return_7d_pct": snapshot.get("return_7d_pct"),
                    "volume_ratio": snapshot.get("volume_ratio"),
                    "rsi_14": snapshot.get("rsi_14"),
                },
            }
            user = json.dumps(context, indent=2)
            output, _ = self.client.complete_model(SYSTEM_PROMPT, user, SentimentOutput)
            if output is not None:
                return AgentVerdict(
                    agent=self.name,
                    model=self.settings.deepseek_model,
                    payload=output.model_dump(),
                    source="llm",
                )
        output = sentiment_fallback(gauge)
        return AgentVerdict(
            agent=self.name,
            model="heuristic-fallback",
            payload=output.model_dump(),
            source="fallback",
        )
