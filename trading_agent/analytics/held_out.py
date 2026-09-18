"""Held-out final validation (V-MONSTER §7) — one-shot, honest by construction.

A fixed 12-month window that is NEVER used for development, tuning,
walk-forward or A/B. It is touched exactly once, at the end of the
implementation, to answer a single question: does the calibration
curve hold out-of-sample?

Locked window (do not edit — the plan's §7 contract): the 12 months
ending at the Phase C model freeze (commit 6e2fd25, 2026-09-17).
All development / walk-forward / A-B windows must end before
HELD_OUT_START.

Verdict rules (no forcing a "ready"):
- INSUFFICIENT_DATA: no confidence bucket reaches the §36 sample
  floor (required_trades for a ±0.10 win-rate deviation).
- FAIL: a sufficient bucket's observed win rate deviates from its
  claimed confidence by more than the detectable-effect floor.
- PASS: at least one sufficient bucket AND none deviates.
The UNVALIDATED labels stay until a PASS — never removed earlier.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from trading_agent.analytics.sample_size import detectable_effect, required_trades
from trading_agent.config import Settings
from trading_agent.store.db import init_engine, session_scope
from trading_agent.store.models import SignalRecord

logger = logging.getLogger(__name__)

# --- The locked window (spec §7 — never retuned, never re-run to "fix") ---
HELD_OUT_END = "2026-09-17"  # Phase C model freeze (commit 6e2fd25)
HELD_OUT_START = "2025-09-17"  # 12 months before the freeze

_BUCKET_WIDTH = 0.05
# §36: N per bucket needed to detect a ±0.10 win-rate deviation
# (power .8, alpha .05, conservative stdev 0.5).
_BUCKET_FLOOR = required_trades(effect_size_r=0.10, stdev_r=0.5)
_GOLD_TICKER = "GC=F"  # (kept for reference — yfinance 15m caps at 60 days)
_DXY_TICKER = "DX-Y.NYB"
_PAXG_SYMBOL = "PAXG/USDT"
_PAXG_EXCHANGE = "binance"  # the only keyless source with 12-month 15m depth

# Data-source honesty (spec §4): yfinance caps 15m at 60 days, so the
# locked 12-month window cannot come from the futures chain. The
# validation therefore runs on the project's own labelled proxy chain —
# PAXG/USDT (24/7 tokenised gold) — with volume axes disabled exactly
# like the live fallback. The report must state this source, never
# present it as institutional XAUUSD.


def _bucket_key(raw: float) -> str:
    return f"{math.floor(raw / _BUCKET_WIDTH) * _BUCKET_WIDTH:.2f}"


def calibration_curve(pairs: list[tuple[float, bool]]) -> dict:
    """Observed win rate vs claimed confidence, per 0.05-wide bucket.

    `pairs` is (raw_confidence, is_win) per RESOLVED trade. The claimed
    win rate of a bucket is the mean raw confidence of its trades — the
    number the calibration (spec §21) says those signals should hit.
    """
    buckets: dict[str, dict] = {}
    for raw, won in pairs:
        key = _bucket_key(raw)
        b = buckets.setdefault(key, {"n": 0, "wins": 0, "sum_raw": 0.0})
        b["n"] += 1
        b["wins"] += 1 if won else 0
        b["sum_raw"] += raw
    for b in buckets.values():
        b["observed"] = b["wins"] / b["n"]
        b["claimed"] = b["sum_raw"] / b["n"]
        b["deviation"] = b["observed"] - b["claimed"]
        b["floor"] = detectable_effect(b["n"], stdev_r=0.5)
        b["sufficient"] = b["n"] >= _BUCKET_FLOOR
    return buckets


def validation_verdict(curve: dict) -> str:
    """PASS / FAIL / INSUFFICIENT_DATA — §7's single honest answer."""
    sufficient = [b for b in curve.values() if b["sufficient"]]
    if not sufficient:
        return "INSUFFICIENT_DATA"
    failing = [b for b in sufficient if abs(b["deviation"]) > b["floor"]]
    return "FAIL" if failing else "PASS"


def _yf_fetch(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=True)
    if data is None or data.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} {start}..{end}")
    df = data.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York")
    df.index = df.index.tz_convert("UTC")
    return df


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    """CSV cache: schema-validated text, never object deserialization."""
    df.to_csv(path)


def _load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def fetch_held_out_frames(cache_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Fetch the locked window exactly once; the cache makes the result
    auditable without ever refetching (the window is never retouched).

    Returns (frames keyed by timeframe, dxy 1h frame) — entry 15m plus
    the snapshot timeframes 1h/4h/1d, with 4h resampled from 1h like
    the live GoldData chain.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if all(
        (cache_dir / f"heldout_{tf}.csv").exists() for tf in ("15m", "1h", "4h", "1d")
    ) and (cache_dir / "heldout_dxy.csv").exists():
        logger.info("held-out cache present — loading without refetching")
        frames = {tf: _load_frame(cache_dir / f"heldout_{tf}.csv") for tf in ("15m", "1h", "4h", "1d")}
        return frames, _load_frame(cache_dir / "heldout_dxy.csv")

    # One coherent proxy series for every gold timeframe (15m paginated,
    # 1h paginated, 4h resampled, 1d paginated) + the real DXY index.
    end_ts = pd.Timestamp(HELD_OUT_END, tz="UTC")
    m15 = _paxg_frame("15m", HELD_OUT_START, HELD_OUT_END)
    m15 = m15[(m15.index >= pd.Timestamp(HELD_OUT_START, tz="UTC")) & (m15.index < end_ts)]
    h1 = _paxg_frame("1h", HELD_OUT_START, HELD_OUT_END)
    h4 = _paxg_frame("4h", HELD_OUT_START, HELD_OUT_END)
    d1 = _paxg_frame("1d", HELD_OUT_START, HELD_OUT_END)
    dxy = _yf_fetch(_DXY_TICKER, HELD_OUT_START, HELD_OUT_END, "60m")

    frames = {"15m": m15, "1h": h1, "4h": h4, "1d": d1}
    for tf, frame in frames.items():
        _save_frame(frame, cache_dir / f"heldout_{tf}.csv")
    _save_frame(dxy, cache_dir / "heldout_dxy.csv")
    return frames, dxy


def _paxg_frame(timeframe: str, start: str, end: str) -> pd.DataFrame:
    """PAXG/USDT OHLCV from the configured exchange, paginated forward.

    The proxy chain (spec §4): tokenised gold, 24/7, keyless. Volume is
    token flow — the pipeline disables every volume axis on it.
    """
    import ccxt

    exchange = getattr(ccxt, _PAXG_EXCHANGE)()
    interval = timeframe if timeframe != "4h" else "1h"  # 4h = resample of 1h
    since_ms = exchange.parse8601(f"{start}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end}T00:00:00Z")
    rows: list[list] = []
    while since_ms < end_ms:
        batch = exchange.fetch_ohlcv(_PAXG_SYMBOL, interval, since=since_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    if not rows:
        raise RuntimeError(f"no {_PAXG_SYMBOL} {interval} data for {start}..{end}")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["ts"]).set_index("ts")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    if timeframe == "4h":
        df = (
            df.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
    return df


def run_held_out_validation(
    settings: Settings,
    frames: dict[str, pd.DataFrame],
    dxy_frames: pd.DataFrame | None,
    db_url: str,
    warmup: int = 500,
) -> dict:
    """The one-shot §7 run: full deterministic pipeline over the locked
    window, then the calibration curve + §36 floors + the verdict.

    Never called twice with the same purpose; the result is the result.
    """
    from trading_agent.backtest.engine import BacktestEngine

    # Reference configuration (no .env), deterministic AI, §24 fill model.
    s = settings.model_copy(
        update={"deepseek_api_key": None, "timeframe": "15m", "backtest_realistic_execution": False}
    )
    init_engine(db_url)
    engine = BacktestEngine(
        s,
        frames,
        symbol="XAUUSD",
        timeframe="15m",
        dxy_frames=dxy_frames,
        start=HELD_OUT_START,
        end=HELD_OUT_END,
        db_url=db_url,
        warmup=warmup,
        # §4 honesty: the window comes from the PAXG proxy chain — token
        # volume is disabled everywhere, exactly like the live fallback.
        volume_basis="proxy",
    )
    report = engine.run()

    # Resolved trades -> (raw confidence, win) via their signal records.
    with session_scope() as session:
        rows = {
            r.signal_id: r
            for r in session.execute(
                SignalRecord.__table__.select()
            ).mappings()
        }
    pairs: list[tuple[float, bool]] = []
    for t in report.trades:
        if t.outcome not in ("WIN", "LOSS"):
            continue
        row = rows.get(t.signal_id)
        raw = (row["fusion"] or {}).get("raw_confidence") if row else None
        if not isinstance(raw, (int, float)):
            continue
        pairs.append((float(raw), t.outcome == "WIN"))

    curve = calibration_curve(pairs)
    verdict = validation_verdict(curve)

    rs = [t.r_multiple for t in report.trades if t.r_multiple is not None]
    expectancy = sum(rs) / len(rs) if rs else None
    stdev = (
        (sum((r - expectancy) ** 2 for r in rs) / (len(rs) - 1)) ** 0.5
        if expectancy is not None and len(rs) > 1
        else None
    )
    claim_floor = (
        required_trades(abs(expectancy), stdev_r=stdev) if expectancy and stdev else None
    )
    stats = report.stats
    return {
        "window_start": HELD_OUT_START,
        "window_end": HELD_OUT_END,
        "data_source": "PAXG/USDT (proxy gold, Binance) + DX-Y.NYB (real DXY)",
        "candles": report.candles,
        "run": {
            "trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats.get("win_rate"),
            "expectancy_r": stats.get("expectancy_r"),
            "profit_factor": stats.get("profit_factor"),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
            "resolved_for_calibration": len(pairs),
        },
        "expectancy_claim": {
            "expectancy_r": expectancy,
            "stdev_r": stdev,
            "required_trades": claim_floor,
            "status": (
                "SUPPORTED"
                if claim_floor is not None and len(rs) >= claim_floor
                else "UNVALIDATED"
            ),
        },
        "calibration": curve,
        "bucket_floor_trades": _BUCKET_FLOOR,
        "verdict": verdict,
    }
