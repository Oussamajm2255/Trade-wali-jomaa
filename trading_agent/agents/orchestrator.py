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
from trading_agent.data.market import MarketDataError
from trading_agent.data.quality import QualityState
from trading_agent.data.snapshot import build_market_snapshot
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


def llm_degraded(verdicts: dict[str, AgentVerdict]) -> bool:
    """True when every available verdict came from the heuristic fallback
    (LLM unreachable — e.g. empty DeepSeek balance)."""
    return bool(verdicts) and all(v.source == "fallback" for v in verdicts.values())


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
        # One coherent snapshot per cycle (spec §3): entry TF + higher TFs,
        # each validated. FAIL stops the cycle before any AI call (§4/§48).
        try:
            snap = build_market_snapshot(self.market, symbol, self.settings, timeframe)
        except MarketDataError as exc:
            logger.error("market data unavailable for %s: %s", symbol, exc)
            return Rejection(symbol=symbol, reason=f"market data unavailable: {exc}"), {}, {}, None
        htf_tf = (
            self.settings.htf_timeframe
            if self.settings.htf_bias_filter_enabled and timeframe != self.settings.htf_timeframe
            else None
        )
        entry = snap.entry_snapshot_for_llm(htf_tf)
        if snap.quality_state == QualityState.FAIL:
            return (
                Rejection(symbol=symbol, reason=f"data quality FAIL: {'; '.join(snap.quality_issues[:3])}"),
                {},
                entry,
                snap.dxy,
            )
        if snap.degraded and not self.settings.data_quality_allow_degraded:
            return (
                Rejection(
                    symbol=symbol,
                    reason=f"data quality DEGRADED (blocked by config): {'; '.join(snap.quality_issues[:3])}",
                ),
                {},
                entry,
                snap.dxy,
            )

        # The bias gate's source is the canonical snapshot (fail-closed above).
        htf_bias = snap.biases.get(self.settings.htf_timeframe) if htf_tf else None
        verdicts = self._run_agents(entry, snap.dxy)
        if not verdicts:
            return Rejection(symbol=symbol, reason="all analysis agents failed"), verdicts, entry, snap.dxy
        side, confidence = self._fuse(verdicts)
        last = snap.candles[timeframe].iloc[-1]
        result = self.risk.evaluate(
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            confidence=confidence,
            price=float(last["close"]),
            atr=float(entry["atr_14"]),
            verdicts=verdicts,
            gauge=snap.dxy,
            htf_bias=htf_bias,
            context=snap.context_for_risk(),
        )
        return result, verdicts, entry, snap.dxy

    def run(self, symbol: str, timeframe: str | None = None) -> SignalProposal | Rejection:
        return self.run_full(symbol, timeframe)[0]
