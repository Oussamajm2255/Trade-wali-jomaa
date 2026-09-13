"""Market-regime agent: trend vs range vs high-volatility classification."""
from __future__ import annotations

import json
import logging

from trading_agent.agents.base import LLMClient
from trading_agent.agents.fallback import regime_fallback
from trading_agent.config import Settings
from trading_agent.schema.types import AgentVerdict, RegimeOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a market-regime agent inside a crypto trading decision pipeline.
You receive trend-strength and volatility metrics for one symbol/timeframe.

Rules:
1. Classify into exactly one regime: "trending_up", "trending_down", "ranging" or "high_volatility".
2. trend_strength is 0.0-1.0 (how strong the trend is; 0 for ranging).
3. Respond with a single JSON object with exactly these keys:
   {"regime": "<one of the four>", "trend_strength": <float 0.0-1.0>, "notes": "2-3 sentences"}
Never invent data."""


class RegimeAgent:
    name = "regime"

    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def analyze(self, snapshot: dict) -> AgentVerdict:
        if self.client.enabled:
            context = {
                "adx_14": snapshot.get("adx_14"),
                "atr_percentile_100": snapshot.get("atr_percentile_100"),
                "ema20_gt_ema50": snapshot.get("ema20_gt_ema50"),
                "ema50_gt_ema200": snapshot.get("ema50_gt_ema200"),
                "macd_hist": snapshot.get("macd_hist"),
                "bb_mid": snapshot.get("bb_mid"),
                "last_close": snapshot.get("last_close"),
            }
            user = json.dumps(context, indent=2)
            output, _ = self.client.complete_model(SYSTEM_PROMPT, user, RegimeOutput)
            if output is not None:
                return AgentVerdict(
                    agent=self.name,
                    model=self.settings.deepseek_model,
                    payload=output.model_dump(),
                    source="llm",
                )
        output = regime_fallback(snapshot)
        return AgentVerdict(
            agent=self.name,
            model="heuristic-fallback",
            payload=output.model_dump(),
            source="fallback",
        )
