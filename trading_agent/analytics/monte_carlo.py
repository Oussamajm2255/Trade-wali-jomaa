"""Monte Carlo risk analysis (spec §28).

Randomizes the order of a completed trade sample (R multiples) and
estimates the distributions of maximum drawdown, longest losing streak
and final equity, plus the empirical risk of ruin. Compounding follows
the live sizing rule (each trade risks `risk_per_trade` of current
equity). This is risk analysis only — never a profitability claim
(spec §28).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloReport:
    n_simulations: int
    seed: int
    risk_of_ruin: float
    max_drawdown_pct: dict[str, float]
    max_losing_streak: dict[str, float]
    final_equity: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "seed": self.seed,
            "risk_of_ruin": self.risk_of_ruin,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_losing_streak": self.max_losing_streak,
            "final_equity": self.final_equity,
        }


def run_monte_carlo(
    trade_rs: list[float],
    n: int = 2000,
    seed: int = 42,
    start_equity: float = 10_000.0,
    risk_per_trade: float = 0.01,
    ruin_pct: float = 0.5,
) -> MonteCarloReport:
    """Simulate `n` randomized orderings of the trade sample (spec §28).

    `ruin_pct` defines ruin: equity falling to this fraction of the
    starting equity (default 50%).
    """
    if not trade_rs:
        raise ValueError("cannot run Monte Carlo on an empty trade sample")
    rng = np.random.default_rng(seed)
    rs = np.asarray(trade_rs, dtype=float)
    drawdowns: list[float] = []
    streaks: list[int] = []
    finals: list[float] = []
    ruined = 0
    for _ in range(n):
        path = rs.copy()
        rng.shuffle(path)
        equity = start_equity
        peak = start_equity
        max_dd = 0.0
        streak = 0
        max_streak = 0
        for r in path:
            equity *= 1.0 + r * risk_per_trade
            if equity > peak:
                peak = equity
            elif peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
            if r < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        drawdowns.append(round(max_dd * 100.0, 4))
        streaks.append(max_streak)
        finals.append(round(equity, 2))
        if equity <= start_equity * ruin_pct:
            ruined += 1

    def percentiles(values: list[float]) -> dict[str, float]:
        return {
            "p5": round(float(np.percentile(values, 5)), 4),
            "p50": round(float(np.percentile(values, 50)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
        }

    return MonteCarloReport(
        n_simulations=n,
        seed=seed,
        risk_of_ruin=round(ruined / n, 4),
        max_drawdown_pct=percentiles(drawdowns),
        max_losing_streak=percentiles([float(s) for s in streaks]),
        final_equity=percentiles(finals),
    )
