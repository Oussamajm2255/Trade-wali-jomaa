"""Parameter sensitivity testing (spec §27).

Replays the same history once per parameter value and looks for STABLE
performance regions — values that stay effective across their neighbours
— instead of chasing the single highest backtest number (spec: do not
choose the parameter that simply produces the highest historical
result). The comparison reuses the deterministic backtest engine, so
every grid point is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_agent.backtest.engine import BacktestEngine
from trading_agent.config import Settings

# Settings fields the spec §27 calls out; the grid is validated against
# this allow-list so a typo can never silently test a non-existent knob.
ALLOWED_PARAMETERS = {
    "min_confidence",
    "setup_quality_min",
    "dxy_long_min",
    "dxy_short_max",
    "atr_stop_mult",
    "take_profit_rr",
}


@dataclass
class SensitivityPoint:
    parameter: str
    value: float
    stats: dict

    def to_dict(self) -> dict:
        return {"parameter": self.parameter, "value": self.value, **self.stats}


@dataclass
class SensitivityReport:
    symbol: str
    timeframe: str
    points: list[SensitivityPoint] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "points": [p.to_dict() for p in self.points],
        }

    def _series(self, metric: str) -> dict[str, list[tuple[float, float | None]]]:
        series: dict[str, list[tuple[float, float | None]]] = {}
        for point in self.points:
            series.setdefault(point.parameter, []).append(
                (point.value, point.stats.get(metric))
            )
        return {p: sorted(v) for p, v in series.items()}

    def stable_regions(
        self,
        metric: str = "expectancy_r",
        tolerance_pct: float = 30.0,
        min_span: int = 2,
    ) -> dict[str, list[dict]]:
        """Contiguous value ranges where the metric stays effective.

        A point is stable when its metric deviates no more than
        `tolerance_pct` from the mean of its neighbours (spec §27:
        prefer parameters that remain effective across nearby values).
        """
        regions: dict[str, list[dict]] = {}
        for parameter, series in self._series(metric).items():
            spans: list[dict] = []
            current: list[tuple[float, float | None]] = []
            for i, (value, value_metric) in enumerate(series):
                if value_metric is None:
                    current = []
                    continue
                neighbours = []
                if i > 0 and series[i - 1][1] is not None:
                    neighbours.append(series[i - 1][1])
                if i + 1 < len(series) and series[i + 1][1] is not None:
                    neighbours.append(series[i + 1][1])
                if not neighbours:
                    stable = True  # single known point: nothing to deviate from
                else:
                    mean = sum(neighbours) / len(neighbours)
                    stable = (
                        abs(value_metric - mean) <= abs(mean) * tolerance_pct / 100.0
                        if mean != 0
                        else abs(value_metric - mean) <= tolerance_pct / 100.0
                    )
                if not stable:
                    current = []
                    continue
                current.append((value, value_metric))
                if len(current) >= min_span:
                    span_values = [v for v, _ in current]
                    span_metrics = [m for _, m in current]
                    # A growing run extends the previous span when the
                    # span's recorded end equals the run's second-to-last
                    # value (the run grows by exactly one point per step).
                    if spans and spans[-1]["end"] == span_values[-2]:
                        spans[-1].update(
                            end=span_values[-1],
                            values=span_values,
                            metric_mean=round(sum(span_metrics) / len(span_metrics), 4),
                        )
                    else:
                        spans.append(
                            {
                                "parameter": parameter,
                                "start": span_values[0],
                                "end": span_values[-1],
                                "values": span_values,
                                "metric_mean": round(sum(span_metrics) / len(span_metrics), 4),
                            }
                        )
            regions[parameter] = spans
        return regions

    def best_stable(self, metric: str = "expectancy_r", **kwargs) -> dict[str, float | None]:
        """Per parameter: the middle of the highest-mean stable region
        (stable first, magnitude second — spec §27), or None."""
        best: dict[str, float | None] = {}
        for parameter, regions in self.stable_regions(metric, **kwargs).items():
            if not regions:
                best[parameter] = None
                continue
            chosen = max(regions, key=lambda r: r["metric_mean"])
            best[parameter] = round((chosen["start"] + chosen["end"]) / 2.0, 6)
        return best


def run_sensitivity(
    settings: Settings,
    frames: dict[str, pd.DataFrame],
    grid: dict[str, list[float]],
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
    dxy_frames: pd.DataFrame | None = None,
    warmup: int | None = None,
    spread_pct: float | None = None,
) -> SensitivityReport:
    """One deterministic replay per grid point; every run is isolated."""
    unknown = set(grid) - ALLOWED_PARAMETERS
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)}; "
            f"allowed: {sorted(ALLOWED_PARAMETERS)}"
        )
    tf = timeframe or settings.timeframe
    report = SensitivityReport(symbol=symbol, timeframe=tf)
    for parameter, values in grid.items():
        for value in values:
            candidate = settings.model_copy(
                update={"deepseek_api_key": None, "timeframe": tf, parameter: value}
            )
            engine = BacktestEngine(
                candidate,
                frames,
                symbol=symbol,
                timeframe=tf,
                dxy_frames=dxy_frames,
                db_url="sqlite:///:memory:",
                warmup=warmup,
                spread_pct=spread_pct,
            )
            run = engine.run()
            report.points.append(SensitivityPoint(parameter, value, run.stats))
    return report
