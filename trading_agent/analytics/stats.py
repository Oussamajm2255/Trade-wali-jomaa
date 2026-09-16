"""Statistical quality (spec §29-§32): regime analytics, conditional
expectancy, minimum sample sizes and the statistical-quality gate input.

Everything here reads RESOLVED signal records (outcome WIN/LOSS with an
R multiple) — the signal database (spec §22) is the historical
population. No statistic is ever manufactured: below the configured
minimum sample the answer is UNKNOWN (spec §31), and the gate never
blocks a signal just because history is missing (spec §32) unless the
historical expectancy of a sufficient sample is explicitly below the
configured floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from trading_agent.store.models import Position, SignalRecord

# Deterministic regime-engine labels -> the spec §29 display taxonomy.
_REGIME_NORMALIZE = {
    "trend_up": "TREND_UP",
    "trend_down": "TREND_DOWN",
    "range": "RANGE",
    "transition": "TRANSITION",
    "high_volatility": "HIGH_VOLATILITY",
    "low_volatility": "LOW_VOLATILITY",
}


@dataclass
class TradeStats:
    """One population's outcome statistics (§29 metric set)."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    expectancy_r: float | None = None
    avg_r: float | None = None
    profit_factor: float | None = None
    total_r: float = 0.0
    max_drawdown_r: float = 0.0

    def to_dict(self) -> dict:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "expectancy_r": self.expectancy_r,
            "avg_r": self.avg_r,
            "profit_factor": self.profit_factor,
            "total_r": self.total_r,
            "max_drawdown_r": self.max_drawdown_r,
        }


def compute_trade_stats(pairs: list[tuple[str, float]]) -> TradeStats:
    """Stats over (outcome, r_multiple) pairs; WIN/LOSS only."""
    stats = TradeStats(trades=len(pairs))
    if not pairs:
        return stats
    stats.wins = sum(1 for outcome, _ in pairs if outcome == "WIN")
    stats.losses = len(pairs) - stats.wins
    stats.win_rate = round(stats.wins / len(pairs), 4)
    rs = [r for _, r in pairs]
    stats.avg_r = stats.expectancy_r = round(sum(rs) / len(rs), 4)
    stats.total_r = round(sum(rs), 4)
    gross_win = sum(r for outcome, r in pairs if outcome == "WIN")
    gross_loss = sum(r for outcome, r in pairs if outcome == "LOSS")
    stats.profit_factor = round(gross_win / abs(gross_loss), 4) if gross_loss < 0 else None
    peak = 0.0
    cum = 0.0
    max_dd = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    stats.max_drawdown_r = round(max_dd, 4)
    return stats


# --- feature extraction from a signal record ------------------------------


def normalize_regime(label: str | None) -> str | None:
    """Map a regime label to the §29 taxonomy (uppercase), or None."""
    if not label:
        return None
    return _REGIME_NORMALIZE.get(str(label).lower()) or str(label).upper()


def side_label(fusion: dict | None) -> str | None:
    score = (fusion or {}).get("direction_score")
    if score is None:
        return None
    return "long" if score > 0 else "short" if score < 0 else None


def regime_label(fusion: dict | None, snap: dict | None) -> str | None:
    label = (fusion or {}).get("regime")
    if not label:
        label = ((snap or {}).get("regime") or {}).get("regime")
    return normalize_regime(label)


def session_label(snap: dict | None) -> str | None:
    return ((snap or {}).get("session_context") or {}).get("session")


def alignment_label(snap: dict | None) -> str | None:
    label = ((snap or {}).get("alignment") or {}).get("label")
    return str(label) if label else None


def structure_labels(snap: dict | None) -> list[str]:
    """SMC features present at signal time (§29): BOS/CHoCH/FVG/sweep."""
    s = (snap or {}).get("structure") or {}
    out: list[str] = []
    if s.get("bos"):
        out.append("BOS")
    if s.get("choch"):
        out.append("CHoCH")
    if s.get("fvgs"):
        out.append("FVG")
    if s.get("sweeps"):
        out.append("LIQUIDITY_SWEEP")
    return out


def dxy_class(side: str | None, snap: dict | None) -> str:
    """DXY supportive / neutral / contradictory / unknown (§29)."""
    if not side:
        return "unknown"
    ctx = (snap or {}).get("dxy_context") or {}
    classification = ctx.get("classification")
    if not classification:
        classification = ((snap or {}).get("dxy_gauge") or {}).get("classification")
    if not classification:
        return "unknown"
    text = str(classification)
    bullish = text.startswith("Bullish")
    bearish = text.startswith("Bearish")
    if side == "long":
        return "supportive" if bullish else "contradictory" if bearish else "neutral"
    return "supportive" if bearish else "contradictory" if bullish else "neutral"


def signal_features(row: SignalRecord) -> dict:
    """The comparable feature set of one historical signal."""
    fusion = row.fusion or {}
    snap = row.market_snapshot or {}
    side = side_label(fusion)
    return {
        "side": side,
        "regime": regime_label(fusion, snap),
        "session": session_label(snap),
        "alignment": alignment_label(snap),
        "structure": structure_labels(snap),
        "dxy": dxy_class(side, snap),
    }


def candidate_features(side: str, regime: str | None, gauge: dict | None) -> dict:
    """The feature set of the CURRENT candidate, built from risk-engine
    inputs so it is directly comparable with signal_features()."""
    return {
        "side": side,
        "regime": normalize_regime(regime),
        "dxy": dxy_class(side, {"dxy_gauge": gauge}),
    }


def _pairs(rows: list[SignalRecord]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for r in rows:
        if r.outcome in ("WIN", "LOSS") and r.r_multiple is not None:
            out.append((r.outcome, float(r.r_multiple)))
    return out


# --- similarity matching ---------------------------------------------------


def _matches(features: dict, conditions: dict) -> bool:
    for key, want in conditions.items():
        got = features.get(key)
        if key == "structure":
            want_list = [want] if isinstance(want, str) else list(want)
            if not all(w in (got or []) for w in want_list):
                return False
            continue
        allowed = [want] if isinstance(want, str) else list(want)
        if got not in allowed:
            return False
    return True


def matching_rows(rows: list[SignalRecord], conditions: dict) -> list[SignalRecord]:
    """Historical signals whose features satisfy every condition."""
    return [r for r in rows if _matches(signal_features(r), conditions)]


# --- §29 regime performance analytics / §30 conditional expectancy ---------


def breakdown(rows: list[SignalRecord], dimension: str) -> dict[str, TradeStats]:
    """Group resolved signals by one feature dimension (§29)."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        features = signal_features(row)
        if dimension == "structure":
            keys = features.get("structure") or ["NONE"]
        else:
            keys = [features.get(dimension)]
        if not keys or keys == [None]:
            keys = ["UNKNOWN"]
        for key in keys:
            groups.setdefault(key, []).append((row.outcome, row.r_multiple))
    return {key: compute_trade_stats(value) for key, value in groups.items()}


def conditional_expectancy(rows: list[SignalRecord], conditions: dict) -> TradeStats:
    """Expectancy of the population matching a condition set (§30)."""
    return compute_trade_stats(_pairs(matching_rows(rows, conditions)))


# --- §31/§32 statistical quality of one candidate -------------------------


@dataclass
class StatisticalQuality:
    """What history says about signals similar to the current candidate."""

    sample_size: int = 0
    historical_win_rate: float | None = None
    historical_expectancy: float | None = None
    average_r: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    min_sample: int = 30

    @property
    def known(self) -> bool:
        return self.sample_size > 0

    @property
    def sufficient(self) -> bool:
        return self.sample_size >= self.min_sample

    def to_dict(self) -> dict:
        return {
            "sample_size": self.sample_size,
            "sufficient": self.sufficient,
            "known": self.known,
            "min_sample": self.min_sample,
            "historical_win_rate": self.historical_win_rate,
            "historical_expectancy": self.historical_expectancy,
            "average_r": self.average_r,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
        }


def statistical_quality(
    rows: list[SignalRecord],
    features: dict,
    min_sample: int = 30,
    positions: dict[str, Position] | None = None,
) -> StatisticalQuality:
    """The similar-signal population for one candidate (spec §32).

    `rows` must already be restricted to resolved signals BEFORE the
    current cycle (the caller enforces the no-look-ahead boundary).
    """
    matched = matching_rows(rows, features)
    stats = compute_trade_stats(_pairs(matched))
    mfe_r: float | None = None
    mae_r: float | None = None
    if positions and matched:
        mfes: list[float] = []
        maes: list[float] = []
        for r in matched:
            if not r.proposal_id:
                continue
            pos = positions.get(r.proposal_id)
            if pos is None or pos.mfe_price is None or pos.mae_price is None:
                continue
            if not pos.entry or not pos.stop or not pos.size:
                continue
            risk = abs(pos.entry - pos.stop) * pos.size
            if risk <= 0:
                continue
            direction = 1.0 if pos.side == "long" else -1.0
            mfes.append((pos.mfe_price - pos.entry) * pos.size * direction / risk)
            maes.append((pos.mae_price - pos.entry) * pos.size * direction / risk)
        if mfes:
            mfe_r = round(sum(mfes) / len(mfes), 4)
        if maes:
            mae_r = round(sum(maes) / len(maes), 4)
    return StatisticalQuality(
        sample_size=stats.trades,
        historical_win_rate=stats.win_rate,
        historical_expectancy=stats.expectancy_r,
        average_r=stats.avg_r,
        mfe_r=mfe_r,
        mae_r=mae_r,
        min_sample=min_sample,
    )


# --- shared query -----------------------------------------------------------


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolved_signals(
    session: Session,
    limit: int = 5000,
    before: datetime | None = None,
    symbol: str | None = None,
) -> list[SignalRecord]:
    """Resolved signal rows (WIN/LOSS), optionally before a timestamp.

    `before` is the no-look-ahead boundary: rows are stored with naive
    UTC timestamps on SQLite, so both sides are normalised to UTC. The
    most recent `limit` rows are kept and returned oldest-first.
    """
    query = select(SignalRecord).where(SignalRecord.outcome.in_(("WIN", "LOSS")))
    if symbol:
        query = query.where(SignalRecord.symbol == symbol)
    rows = list(session.scalars(query.order_by(SignalRecord.ts.desc()).limit(limit)))
    if before is not None:
        boundary = _utc(before)
        rows = [r for r in rows if _utc(r.ts) < boundary]
    rows.reverse()
    return rows
