"""A/B strategy comparison (spec §40/§41): LEGACY_BASELINE vs
INTELLIGENCE_V2 on identical historical data, with a verdict that never
rewards fewer trades.

The baseline is the git-tagged pre-upgrade pipeline (`LEGACY_BASELINE`,
tag on the last pre-upgrade commit). It is replayed here as a
deterministic gate PROFILE: the current pipeline with every
post-upgrade gate disabled — reproducing the tagged decision path
(AI signal -> v1 filters -> hard risk engine). INTELLIGENCE_V2 runs
the full gate stack. Both sides share identical candles, spread,
warmup and time range (spec §40).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_agent.backtest.engine import BacktestEngine, BacktestReport
from trading_agent.config import Settings

STRATEGY_NAMES = ("legacy_baseline", "intelligence_v2")

# Post-upgrade gates disabled for the baseline profile. The v1 decision
# path — AI signal gate, DXY filter, HTF bias, min_confidence and the
# hard risk engine — stays identical in both profiles.
STRATEGY_PROFILES: dict[str, dict] = {
    "legacy_baseline": {
        "setup_quality_min": 0.0,
        "conflict_block_conflicted": False,
        "require_statistical_edge": False,
        "statistical_quality_enabled": False,
        "no_trade_high_volatility": False,
    },
    "intelligence_v2": {},
}


def _sibling_db(url: str, name: str) -> str:
    """One isolated database per strategy: the runs must never share
    risk state or signal records."""
    if url.startswith("sqlite:///") and "?" not in url:
        path = url.removeprefix("sqlite:///")
        stem, dot, ext = path.rpartition(".")
        if stem:
            return f"sqlite:///{stem}_{name}{dot}{ext}"
    return f"{url}_{name}"


def _reset_db(url: str) -> None:
    """Drop and recreate one comparison database so repeated runs are
    deterministic (spec §25): fresh risk state, no stale records."""
    from sqlalchemy import create_engine

    from trading_agent.store.models import Base

    engine = create_engine(url, future=True)
    Base.metadata.drop_all(engine)
    engine.dispose()


def _verdict(a: dict, b: dict, min_trades: int) -> tuple[str, str]:
    """Verdict of INTELLIGENCE_V2 (b) against LEGACY_BASELINE (a).

    §41: improved only on genuinely better risk-adjusted quality —
    expectancy AND drawdown not worse AND profit factor not worse.
    Generating fewer trades alone is never an improvement.
    """
    fewer_trades_note = ""
    if b.get("trades", 0) < a.get("trades", 0):
        fewer_trades_note = " (fewer trades alone is not an improvement — §41)"
    if (a.get("trades") or 0) < min_trades or (b.get("trades") or 0) < min_trades:
        return "INSUFFICIENT_DATA", (
            f"need at least {min_trades} resolved trades per side"
            + fewer_trades_note
        )
    ea, eb = a["expectancy_r"], b["expectancy_r"]
    if ea is None or eb is None:
        return "INSUFFICIENT_DATA", (
            "both sides need resolved trades to compare" + fewer_trades_note
        )
    da, db = a["max_drawdown_pct"], b["max_drawdown_pct"]
    pa, pb = a["profit_factor"], b["profit_factor"]
    dd_ok = db <= da * 1.1  # drawdown not meaningfully worse
    pf_ok = pa is None or pb is None or pb >= pa * 0.9  # PF not worse
    if eb > ea and dd_ok and pf_ok:
        return "IMPROVED", (
            "better expectancy with drawdown and profit factor not worse"
        )
    dd_worse = db >= da
    pf_worse = pa is not None and pb is not None and pb < pa
    if eb < ea and (dd_worse or pf_worse):
        return "WORSE", (
            "lower expectancy while drawdown/profit factor did not improve"
            + fewer_trades_note
        )
    return "MIXED", (
        "no clear improvement across expectancy, drawdown and profit factor"
        + fewer_trades_note
    )


@dataclass
class ABResult:
    """One side of the comparison plus its verdict against the other."""

    strategy: str
    report: BacktestReport
    verdict: str = ""
    verdict_note: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "stats": self.report.stats,
            "db_url": self.report.db_url,
            "verdict": self.verdict,
            "verdict_note": self.verdict_note,
        }


@dataclass
class ABReport:
    """§41 comparison report: both runs + the shared verdict."""

    symbol: str
    timeframe: str
    start_ts: str
    end_ts: str
    results: dict[str, ABResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "results": {name: r.to_dict() for name, r in self.results.items()},
        }


def run_comparison(
    settings: Settings,
    frames: dict,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
    dxy_frames=None,
    start: str | None = None,
    end: str | None = None,
    db_url: str | None = None,
    spread_pct: float | None = None,
    warmup: int | None = None,
    min_trades: int = 10,
) -> ABReport:
    """Run both strategies on identical data and produce the §41 report."""
    tf = timeframe or settings.timeframe
    base_db = db_url or settings.backtest_db_url
    reports: dict[str, BacktestReport] = {}
    for name in STRATEGY_NAMES:
        profiled = settings.model_copy(update=dict(STRATEGY_PROFILES.get(name, {})))
        url = _sibling_db(base_db, name)
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
        reports[name] = engine.run()
    a = reports["legacy_baseline"]
    b = reports["intelligence_v2"]
    verdict, note = _verdict(a.stats, b.stats, min_trades)
    return ABReport(
        symbol=symbol,
        timeframe=tf,
        start_ts=b.start_ts,
        end_ts=b.end_ts,
        results={
            "legacy_baseline": ABResult(
                "legacy_baseline", a, verdict="BASELINE", verdict_note="reference run"
            ),
            "intelligence_v2": ABResult(
                "intelligence_v2", b, verdict=verdict, verdict_note=note
            ),
        },
    )
