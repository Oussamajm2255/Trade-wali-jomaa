"""Feature ablation (V-MONSTER §72): WITH vs WITHOUT each feature group.

Each group is removed from the pipeline through its snapshot-level
toggle (`feature_*_enabled`, consumed in `build_market_snapshot`): the
group's inputs vanish and every downstream module answers its documented
neutral — no data never blocks, so this is the honest equivalent of a
real ablation study without re-implementing the pipeline.

The harness replays the SAME window once per group (the window geometry
`run_walk_forward` uses, spec §40) on an isolated database per run and
applies the §41 verdict rules (expectancy + drawdown + profit factor;
fewer trades alone is never an improvement). Verdict direction:
IMPROVED means the WITHOUT run beat the WITH baseline — the group cost
more than it gave on this window; WORSE means removing it hurt, i.e.
the group earned its place; MIXED/INSUFFICIENT_DATA are no conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_agent.analytics.compare import _reset_db, _sibling_db, _verdict
from trading_agent.backtest.engine import BacktestEngine, BacktestReport
from trading_agent.config import Settings

# Group -> the settings profile that removes the group's snapshot inputs.
FEATURE_GROUPS: dict[str, dict] = {
    "smc": {"feature_smc_enabled": False},
    "dxy": {"feature_dxy_enabled": False},
    "vwap": {"feature_vwap_enabled": False},
    "liquidity": {"feature_liquidity_enabled": False},
    "speed": {"feature_speed_enabled": False},
}


@dataclass
class AblationResult:
    """One WITHOUT run plus its §41 verdict against the WITH baseline."""

    group: str
    report: BacktestReport
    verdict: str = ""
    verdict_note: str = ""

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "stats": self.report.stats,
            "db_url": self.report.db_url,
            "verdict": self.verdict,
            "verdict_note": self.verdict_note,
        }


@dataclass
class AblationReport:
    """§72 report: the WITH baseline plus one WITHOUT run per group."""

    symbol: str
    timeframe: str
    start_ts: str
    end_ts: str
    baseline: AblationResult | None = None
    results: dict[str, AblationResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "results": {name: r.to_dict() for name, r in self.results.items()},
        }


def run_ablation(
    settings: Settings,
    frames: dict,
    groups: list[str] | None = None,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
    dxy_frames=None,
    start: str | None = None,
    end: str | None = None,
    db_url: str | None = None,
    spread_pct: float | None = None,
    warmup: int | None = None,
    min_trades: int = 10,
) -> AblationReport:
    """Run WITH (the caller's settings) and WITHOUT (each group off).

    Identical candles, spread, warmup and time range on every run; the
    caller's settings ARE the all-features-on profile, so the baseline
    is exactly what a plain backtest would produce. Each run gets its
    own isolated database (fresh risk state, no shared records).
    """
    tf = timeframe or settings.timeframe
    base_db = db_url or settings.backtest_db_url
    groups = list(FEATURE_GROUPS) if groups is None else groups
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}")

    def _run(profiled: Settings, url: str) -> BacktestReport:
        _reset_db(url)
        engine = BacktestEngine(
            settings=profiled,
            frames=frames,
            symbol=symbol,
            timeframe=tf,
            dxy_frames=dxy_frames,
            start=start,
            end=end,
            db_url=url,
            spread_pct=spread_pct,
            warmup=warmup,
        )
        return engine.run()

    # The caller's own database is never touched: the baseline gets its
    # sibling too (compare.py convention, spec §40 isolation).
    base = _run(settings, _sibling_db(base_db, "abl_with"))
    report = AblationReport(
        symbol=symbol,
        timeframe=tf,
        start_ts=base.start_ts,
        end_ts=base.end_ts,
        baseline=AblationResult("with_all_features", base, "BASELINE", "reference run"),
    )
    for group in groups:
        profiled = settings.model_copy(update=dict(FEATURE_GROUPS[group]))
        run = _run(profiled, _sibling_db(base_db, f"abl_{group}"))
        verdict, note = _verdict(base.stats, run.stats, min_trades)
        report.results[group] = AblationResult(group, run, verdict, note)
    return report
