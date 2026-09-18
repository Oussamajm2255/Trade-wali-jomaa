"""Sample-size derivation (V-MONSTER §36, Phase L).

No fixed "100 trades" rule: the required N for any performance claim
is derived from effect size, variance and power. `required_trades`
solves the classic two-sided power equation with the normal
approximation (z quantiles); `detectable_effect` inverts it — the
smallest expectancy difference a given sample can support. Both are
pure math, deterministic, and used by every downstream claim gate
(compare.py-style verdicts stay INSUFFICIENT_DATA until these N's
hold, spec §41).
"""

from __future__ import annotations

import math
from statistics import NormalDist

_Z = NormalDist().inv_cdf


def _validate(effect_size_r: float, stdev_r: float | None, power: float, alpha: float) -> float:
    if effect_size_r <= 0:
        raise ValueError("effect_size_r must be positive")
    if stdev_r is not None and stdev_r <= 0:
        raise ValueError("stdev_r must be positive when provided")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    # stdev None -> the effect size is already expressed in stdev units.
    return stdev_r if stdev_r is not None else 1.0


def required_trades(
    effect_size_r: float,
    stdev_r: float | None = None,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Trades needed per side to detect `effect_size_r` at `power`.

    The standard normal approximation of the two-sided t-test sample
    size: N ~= 2 (z_{1-alpha/2} + z_power)^2 (stdev/effect)^2, rounded
    up. One-sample claims (expectancy vs 0) and two-sample comparisons
    (A/B feature ablation) share the formula; the caller supplies the
    effect/stdev pair that matches the claim.
    """
    sd = _validate(effect_size_r, stdev_r, power, alpha)
    z = _Z(1.0 - alpha / 2.0) + _Z(power)
    return math.ceil(2.0 * z * z * (sd / effect_size_r) ** 2)


def detectable_effect(
    n_trades: int,
    stdev_r: float | None = None,
    *,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Smallest expectancy difference `n_trades` can detect, in R units.

    The inverse of `required_trades`: an observed effect below this
    floor is indistinguishable from noise at this sample size, so the
    claim must be labeled UNVALIDATED — never presented as evidence.
    """
    if n_trades <= 0:
        raise ValueError("n_trades must be positive")
    if stdev_r is not None and stdev_r <= 0:
        raise ValueError("stdev_r must be positive when provided")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    sd = stdev_r if stdev_r is not None else 1.0
    z = _Z(1.0 - alpha / 2.0) + _Z(power)
    return sd * math.sqrt(2.0 * z * z / n_trades)
