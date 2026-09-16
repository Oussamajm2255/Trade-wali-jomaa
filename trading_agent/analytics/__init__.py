"""Statistical analytics for INTELLIGENCE_V2 (spec §26-§32).

Walk-forward validation, parameter sensitivity, Monte Carlo risk
analysis and the statistical-quality machinery. Every report is
computed from resolved signal/position history or deterministic
backtest replays — never manufactured.
"""

from trading_agent.analytics.monte_carlo import MonteCarloReport, run_monte_carlo
from trading_agent.analytics.sensitivity import (
    ALLOWED_PARAMETERS,
    SensitivityPoint,
    SensitivityReport,
    run_sensitivity,
)
from trading_agent.analytics.stats import (
    StatisticalQuality,
    TradeStats,
    breakdown,
    candidate_features,
    compute_trade_stats,
    conditional_expectancy,
    dxy_class,
    matching_rows,
    normalize_regime,
    resolved_signals,
    signal_features,
    statistical_quality,
)
from trading_agent.analytics.walk_forward import (
    WalkForwardConfig,
    WalkForwardReport,
    WalkForwardWindow,
    run_walk_forward,
)

__all__ = [
    "ALLOWED_PARAMETERS",
    "MonteCarloReport",
    "SensitivityPoint",
    "SensitivityReport",
    "StatisticalQuality",
    "TradeStats",
    "WalkForwardConfig",
    "WalkForwardReport",
    "WalkForwardWindow",
    "breakdown",
    "candidate_features",
    "compute_trade_stats",
    "conditional_expectancy",
    "dxy_class",
    "matching_rows",
    "normalize_regime",
    "resolved_signals",
    "run_monte_carlo",
    "run_sensitivity",
    "run_walk_forward",
    "signal_features",
    "statistical_quality",
]
