"""Walk-forward validation (spec §26): TRAIN -> VALIDATION -> OUT-OF-SAMPLE.

The robot's parameters are fixed by configuration — the risk engine
never optimizes from data, so walk-forward here measures the STABILITY
of a fixed strategy across consecutive unseen periods: each test window
is replayed on a fresh, isolated database whose indicator warmup is the
tail of the preceding train segment. No test candle ever influences any
other window's indicators, calibration buckets or equity state, and the
calibration inside one window only sees that window's own resolved
signals (spec: test data never influences parameter optimization).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trading_agent.backtest.engine import BacktestEngine
from trading_agent.config import Settings


@dataclass
class WalkForwardConfig:
    """Window geometry, in candles of the entry timeframe."""

    train_bars: int = 1000
    val_bars: int = 0
    test_bars: int = 500
    step_bars: int | None = None  # default: test_bars (non-overlapping)
    min_test_bars: int = 1

    def __post_init__(self) -> None:
        if self.step_bars is None:
            self.step_bars = self.test_bars
        if self.train_bars < 1 or self.test_bars < self.min_test_bars or self.step_bars < 1:
            raise ValueError("invalid walk-forward window geometry")
        if self.val_bars < 0:
            raise ValueError("val_bars cannot be negative")


@dataclass
class WalkForwardWindow:
    train_start: str
    train_end: str
    val_start: str | None
    val_end: str | None
    test_start: str
    test_end: str
    candles: int
    stats: dict

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "candles": self.candles,
            "stats": self.stats,
        }


@dataclass
class WalkForwardReport:
    symbol: str
    timeframe: str
    config: WalkForwardConfig
    windows: list[WalkForwardWindow] = field(default_factory=list)

    @property
    def aggregate(self) -> dict:
        """Stability summary across the out-of-sample windows."""
        expectancy = [
            w.stats["expectancy_r"]
            for w in self.windows
            if w.stats.get("expectancy_r") is not None
        ]
        win_rates = [
            w.stats["win_rate"] for w in self.windows if w.stats.get("win_rate") is not None
        ]
        profitable = sum(1 for w in self.windows if (w.stats.get("total_r") or 0) > 0)
        avg = sum(expectancy) / len(expectancy) if expectancy else None
        variance = (
            sum((e - avg) ** 2 for e in expectancy) / len(expectancy) if expectancy else None
        )
        return {
            "windows": len(self.windows),
            "total_trades": sum(w.stats.get("trades", 0) for w in self.windows),
            "profitable_windows": profitable,
            "profitable_ratio": round(profitable / len(self.windows), 4) if self.windows else None,
            "avg_expectancy_r": round(avg, 4) if avg is not None else None,
            "median_expectancy_r": (
                round(sorted(expectancy)[len(expectancy) // 2], 4) if expectancy else None
            ),
            "stdev_expectancy_r": round(variance ** 0.5, 4) if variance is not None else None,
            "avg_win_rate": round(sum(win_rates) / len(win_rates), 4) if win_rates else None,
        }

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "config": {
                "train_bars": self.config.train_bars,
                "val_bars": self.config.val_bars,
                "test_bars": self.config.test_bars,
                "step_bars": self.config.step_bars,
            },
            "aggregate": self.aggregate,
            "windows": [w.to_dict() for w in self.windows],
        }


def run_walk_forward(
    settings: Settings,
    frames: dict[str, pd.DataFrame],
    config: WalkForwardConfig,
    symbol: str = "XAUUSD",
    timeframe: str | None = None,
    dxy_frames: pd.DataFrame | None = None,
    warmup: int | None = None,
) -> WalkForwardReport:
    """Replay consecutive out-of-sample windows of a fixed configuration.

    Each window gets its own in-memory database (fresh risk state, fresh
    calibration) and a frame whose first `warmup` candles are the tail of
    the preceding train/validation data — indicators warm up on real
    history without a single analyzed candle leaking across windows.
    """
    tf = timeframe or settings.timeframe
    entry = frames.get(tf)
    if entry is None or entry.empty:
        raise ValueError(f"no historical frame for {symbol} {tf}")
    entry = entry[~entry.index.duplicated(keep="last")].sort_index()
    warmup = warmup if warmup is not None else settings.ohlcv_limit
    if config.train_bars < warmup:
        raise ValueError(
            f"train_bars ({config.train_bars}) must cover the warmup ({warmup})"
        )
    first_test = warmup + config.train_bars + config.val_bars
    if len(entry) < first_test + config.test_bars:
        raise ValueError(
            f"need at least {first_test + config.test_bars} candles for one window, "
            f"frame has {len(entry)}"
        )

    report = WalkForwardReport(symbol=symbol, timeframe=tf, config=config)
    for offset in range(0, len(entry) - first_test - config.test_bars + 1, config.step_bars):
        test_start_idx = first_test + offset
        test_end_idx = test_start_idx + config.test_bars
        window_df = entry.iloc[test_start_idx - warmup : test_end_idx]
        engine = BacktestEngine(
            settings,
            {tf: window_df},
            symbol=symbol,
            timeframe=tf,
            dxy_frames=dxy_frames,
            db_url="sqlite:///:memory:",
            warmup=warmup,
        )
        run = engine.run()
        val_start_idx = test_start_idx - config.val_bars
        train_start_idx = val_start_idx - config.train_bars
        report.windows.append(
            WalkForwardWindow(
                train_start=str(entry.index[max(0, train_start_idx)]),
                train_end=str(entry.index[val_start_idx - 1]),
                val_start=str(entry.index[val_start_idx]) if config.val_bars else None,
                val_end=str(entry.index[test_start_idx - 1]) if config.val_bars else None,
                test_start=str(entry.index[test_start_idx]),
                test_end=str(entry.index[test_end_idx - 1]),
                candles=run.candles,
                stats=run.stats,
            )
        )
    return report
