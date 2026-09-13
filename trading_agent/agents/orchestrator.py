"""Orchestrator: runs the analysis agents, fuses verdicts into a scored
signal, then hands it to the risk engine (which has final say)."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from trading_agent.agents.base import LLMClient
from trading_agent.agents.regime import RegimeAgent
from trading_agent.agents.sentiment import SentimentAgent
from trading_agent.agents.technical import TechnicalAgent
from trading_agent.config import Settings
from trading_agent.data.bias import compute_htf_bias
from trading_agent.data.market import MarketData
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import (
    AgentVerdict,
    Bias,
    Regime,
    Rejection,
    Side,
    SignalProposal,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, settings: Settings, market: MarketData, risk: RiskEngine) -> None:
        self.settings = settings
        self.market = market
        self.risk = risk
        self.llm = LLMClient(settings)
        self.technical = TechnicalAgent(self.llm, settings)
        self.sentiment = SentimentAgent(self.llm, settings)
        self.regime = RegimeAgent(self.llm, settings)

    def _run_agents(self, snapshot: dict, gauge: dict | None) -> dict[str, AgentVerdict]:
        tasks: dict[str, Callable[[], AgentVerdict]] = {
            "technical": lambda: self.technical.analyze(snapshot),
            "sentiment": lambda: self.sentiment.analyze(gauge, snapshot),
            "regime": lambda: self.regime.analyze(snapshot),
        }
        verdicts: dict[str, AgentVerdict] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    verdicts[name] = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate agent crashes
                    logger.error("agent %s crashed: %s", name, exc)
        return verdicts

    def _fuse(self, verdicts: dict[str, AgentVerdict]) -> tuple[Side, float]:
        """Weighted fusion of agent verdicts into (side, confidence)."""
        score = 0.0
        for name, verdict in verdicts.items():
            payload = verdict.payload
            if name == "technical":
                bias = Bias(payload["bias"])
                sign = 1.0 if bias == Bias.LONG else -1.0 if bias == Bias.SHORT else 0.0
                score += self.settings.weight_technical * sign * payload["conviction"]
            elif name == "sentiment":
                score += self.settings.weight_sentiment * payload["score"]
            else:  # regime
                regime = Regime(payload["regime"])
                sign = 1.0 if regime == Regime.TRENDING_UP else -1.0 if regime == Regime.TRENDING_DOWN else 0.0
                score += self.settings.weight_regime * sign * payload["trend_strength"]
        threshold = self.settings.side_threshold
        if score >= threshold:
            side = Side.LONG
        elif score <= -threshold:
            side = Side.SHORT
        else:
            side = Side.NEUTRAL
        return side, min(1.0, abs(score))

    def run_full(
        self, symbol: str, timeframe: str | None = None
    ) -> tuple[SignalProposal | Rejection, dict[str, AgentVerdict], dict, dict | None]:
        """Full pipeline; also returns verdicts/snapshot/gauge for reporting."""
        timeframe = timeframe or self.settings.timeframe
        df, snapshot, gauge = self.market.analysis_input(symbol, timeframe, self.settings.ohlcv_limit)

        # Pro multi-timeframe: the entry TF spots triggers, the HTF sets the
        # directional bias (deterministic, computed locally). The bias is both
        # shown to the LLM (context) and enforced by the risk engine (hard gate).
        htf_bias: dict | None = None
        if self.settings.htf_bias_filter_enabled and timeframe != self.settings.htf_timeframe:
            try:
                htf_df = self.market.fetch_ohlcv(
                    symbol, self.settings.htf_timeframe, self.settings.ohlcv_limit
                )
                htf_bias = compute_htf_bias(htf_df, self.settings.htf_adx_min)
                snapshot["htf_bias"] = {
                    "timeframe": self.settings.htf_timeframe,
                    "bias": htf_bias["bias"],
                    "detail": htf_bias["detail"],
                    "adx": htf_bias["adx"],
                }
            except Exception as exc:  # noqa: BLE001 - fail closed below
                logger.error("HTF bias unavailable for %s: %s", symbol, exc)
                return (
                    Rejection(symbol=symbol, reason=f"HTF bias unavailable: {exc}"),
                    {},
                    snapshot,
                    gauge,
                )

        verdicts = self._run_agents(snapshot, gauge)
        if not verdicts:
            return Rejection(symbol=symbol, reason="all analysis agents failed"), verdicts, snapshot, gauge
        side, confidence = self._fuse(verdicts)
        last = df.iloc[-1]
        result = self.risk.evaluate(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            confidence=confidence,
            price=float(last["close"]),
            atr=float(snapshot["atr_14"]),
            verdicts=verdicts,
            gauge=gauge,
            htf_bias=htf_bias,
        )
        return result, verdicts, snapshot, gauge

    def run(self, symbol: str, timeframe: str | None = None) -> SignalProposal | Rejection:
        return self.run_full(symbol, timeframe)[0]
